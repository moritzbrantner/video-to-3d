use crate::Point3;
use serde::Serialize;

const MIN_ALIGNMENT_ANCHORS: usize = 12;
const AGREEMENT_RELATIVE_ERROR: f32 = 0.15;
const TRIM_FRACTION: f32 = 0.80;

#[derive(Clone, Copy, Debug)]
pub struct LearnedDepthCamera {
    pub frame_index: usize,
    pub rotation: [f32; 9],
    pub translation: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct RelativeDepthFrame<'a> {
    pub frame_index: usize,
    pub width: usize,
    pub height: usize,
    pub values: &'a [f32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelativeDepthFitKind {
    Direct,
    Inverse,
}

#[derive(Clone, Debug, Serialize)]
pub struct RelativeDepthFrameDiagnostics {
    pub frame_index: usize,
    pub projected_geometry_points: usize,
    pub valid_alignment_anchors: usize,
    pub fit_kind: Option<RelativeDepthFitKind>,
    pub scale: Option<f32>,
    pub offset: Option<f32>,
    pub median_relative_error: Option<f32>,
    pub agreement_ratio_15_percent: Option<f32>,
    pub calibration_usable: bool,
    pub skip_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RelativeDepthEvaluation {
    pub agreement_threshold: f32,
    pub frames: Vec<RelativeDepthFrameDiagnostics>,
}

#[derive(Clone, Copy, Debug)]
struct Anchor {
    learned: f32,
    geometric_depth: f32,
}

#[derive(Clone, Copy, Debug)]
struct LinearFit {
    kind: RelativeDepthFitKind,
    scale: f32,
    offset: f32,
    median_relative_error: f32,
    agreement_ratio: f32,
}

pub fn evaluate_relative_depth(
    cameras: &[LearnedDepthCamera],
    dense_points: &[Point3],
    focal_pixels: f32,
    frames: &[RelativeDepthFrame<'_>],
) -> RelativeDepthEvaluation {
    let diagnostics = frames
        .iter()
        .map(|frame| evaluate_frame(cameras, dense_points, focal_pixels, *frame))
        .collect();
    RelativeDepthEvaluation {
        agreement_threshold: AGREEMENT_RELATIVE_ERROR,
        frames: diagnostics,
    }
}

fn evaluate_frame(
    cameras: &[LearnedDepthCamera],
    dense_points: &[Point3],
    focal_pixels: f32,
    frame: RelativeDepthFrame<'_>,
) -> RelativeDepthFrameDiagnostics {
    if frame.width == 0 || frame.height == 0 || frame.values.len() != frame.width * frame.height {
        return skipped(
            frame.frame_index,
            "relative-depth dimensions are inconsistent",
        );
    }
    if !focal_pixels.is_finite() || focal_pixels <= 0.0 {
        return skipped(frame.frame_index, "accepted focal length is unavailable");
    }
    let Some(camera) = cameras
        .iter()
        .find(|camera| camera.frame_index == frame.frame_index)
    else {
        return skipped(
            frame.frame_index,
            "relative depth does not correspond to an accepted final camera",
        );
    };

    let mut projected_geometry_points = 0;
    let mut anchors = Vec::new();
    for point in dense_points {
        let Some((u, v, depth)) = project(camera, point, frame.width, frame.height, focal_pixels)
        else {
            continue;
        };
        projected_geometry_points += 1;
        let Some(learned) = sample_nearest(frame, u, v) else {
            continue;
        };
        if learned.is_finite() && learned > 1.0e-6 && depth.is_finite() && depth > 1.0e-6 {
            anchors.push(Anchor {
                learned,
                geometric_depth: depth,
            });
        }
    }

    if anchors.len() < MIN_ALIGNMENT_ANCHORS {
        return RelativeDepthFrameDiagnostics {
            frame_index: frame.frame_index,
            projected_geometry_points,
            valid_alignment_anchors: anchors.len(),
            fit_kind: None,
            scale: None,
            offset: None,
            median_relative_error: None,
            agreement_ratio_15_percent: None,
            calibration_usable: false,
            skip_reason: Some(format!(
                "fewer than {MIN_ALIGNMENT_ANCHORS} valid geometric alignment anchors"
            )),
        };
    }

    let direct = robust_linear_fit(&anchors, RelativeDepthFitKind::Direct);
    let inverse = robust_linear_fit(&anchors, RelativeDepthFitKind::Inverse);
    let best = match (direct, inverse) {
        (Some(left), Some(right)) => Some(
            if left.median_relative_error <= right.median_relative_error {
                left
            } else {
                right
            },
        ),
        (Some(fit), None) | (None, Some(fit)) => Some(fit),
        (None, None) => None,
    };

    let Some(fit) = best else {
        return RelativeDepthFrameDiagnostics {
            frame_index: frame.frame_index,
            projected_geometry_points,
            valid_alignment_anchors: anchors.len(),
            fit_kind: None,
            scale: None,
            offset: None,
            median_relative_error: None,
            agreement_ratio_15_percent: None,
            calibration_usable: false,
            skip_reason: Some("relative-depth alignment is numerically degenerate".into()),
        };
    };

    let calibration_usable =
        fit.median_relative_error <= AGREEMENT_RELATIVE_ERROR && fit.agreement_ratio >= 0.60;
    RelativeDepthFrameDiagnostics {
        frame_index: frame.frame_index,
        projected_geometry_points,
        valid_alignment_anchors: anchors.len(),
        fit_kind: Some(fit.kind),
        scale: Some(fit.scale),
        offset: Some(fit.offset),
        median_relative_error: Some(fit.median_relative_error),
        agreement_ratio_15_percent: Some(fit.agreement_ratio),
        calibration_usable,
        skip_reason: (!calibration_usable).then(|| {
            "learned depth does not agree closely enough with accepted geometric depth".into()
        }),
    }
}

fn skipped(frame_index: usize, reason: &str) -> RelativeDepthFrameDiagnostics {
    RelativeDepthFrameDiagnostics {
        frame_index,
        projected_geometry_points: 0,
        valid_alignment_anchors: 0,
        fit_kind: None,
        scale: None,
        offset: None,
        median_relative_error: None,
        agreement_ratio_15_percent: None,
        calibration_usable: false,
        skip_reason: Some(reason.into()),
    }
}

fn project(
    camera: &LearnedDepthCamera,
    point: &Point3,
    width: usize,
    height: usize,
    focal: f32,
) -> Option<(f32, f32, f32)> {
    let x = camera.rotation[0] * point.x
        + camera.rotation[1] * point.y
        + camera.rotation[2] * point.z
        + camera.translation[0];
    let y = camera.rotation[3] * point.x
        + camera.rotation[4] * point.y
        + camera.rotation[5] * point.z
        + camera.translation[1];
    let z = camera.rotation[6] * point.x
        + camera.rotation[7] * point.y
        + camera.rotation[8] * point.z
        + camera.translation[2];
    if !x.is_finite() || !y.is_finite() || !z.is_finite() || z <= 1.0e-6 {
        return None;
    }
    let u = focal * x / z + width as f32 * 0.5;
    let v = focal * y / z + height as f32 * 0.5;
    if u < 0.0 || v < 0.0 || u >= width as f32 || v >= height as f32 {
        return None;
    }
    Some((u, v, z))
}

fn sample_nearest(frame: RelativeDepthFrame<'_>, u: f32, v: f32) -> Option<f32> {
    let x = (u.round() as isize).clamp(0, frame.width as isize - 1) as usize;
    let y = (v.round() as isize).clamp(0, frame.height as isize - 1) as usize;
    frame.values.get(y * frame.width + x).copied()
}

fn robust_linear_fit(anchors: &[Anchor], kind: RelativeDepthFitKind) -> Option<LinearFit> {
    let initial = least_squares(anchors, kind)?;
    let mut ranked: Vec<(usize, f32)> = anchors
        .iter()
        .enumerate()
        .map(|(index, anchor)| (index, relative_error(initial, *anchor, kind)))
        .filter(|(_, error)| error.is_finite())
        .collect();
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
    let keep = ((ranked.len() as f32 * TRIM_FRACTION).ceil() as usize)
        .max(MIN_ALIGNMENT_ANCHORS)
        .min(ranked.len());
    let trimmed: Vec<Anchor> = ranked
        .into_iter()
        .take(keep)
        .map(|(index, _)| anchors[index])
        .collect();
    let refined = least_squares(&trimmed, kind)?;

    let mut errors: Vec<f32> = anchors
        .iter()
        .map(|anchor| relative_error(refined, *anchor, kind))
        .filter(|error| error.is_finite())
        .collect();
    if errors.is_empty() {
        return None;
    }
    errors.sort_by(f32::total_cmp);
    let median_relative_error = median_sorted(&errors);
    let agreement_ratio = errors
        .iter()
        .filter(|error| **error <= AGREEMENT_RELATIVE_ERROR)
        .count() as f32
        / errors.len() as f32;
    Some(LinearFit {
        kind,
        scale: refined.0,
        offset: refined.1,
        median_relative_error,
        agreement_ratio,
    })
}

fn least_squares(anchors: &[Anchor], kind: RelativeDepthFitKind) -> Option<(f32, f32)> {
    if anchors.len() < 2 {
        return None;
    }
    let transform = |value: f32| match kind {
        RelativeDepthFitKind::Direct => value,
        RelativeDepthFitKind::Inverse => 1.0 / value,
    };
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut count = 0.0f64;
    for anchor in anchors {
        let x = transform(anchor.learned) as f64;
        let y = anchor.geometric_depth as f64;
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        count += 1.0;
    }
    if count < 2.0 {
        return None;
    }
    let denominator = count * sum_xx - sum_x * sum_x;
    if denominator.abs() <= 1.0e-12 {
        return None;
    }
    let scale = ((count * sum_xy - sum_x * sum_y) / denominator) as f32;
    let offset = ((sum_y - scale as f64 * sum_x) / count) as f32;
    if scale.is_finite() && offset.is_finite() {
        Some((scale, offset))
    } else {
        None
    }
}

fn relative_error(fit: (f32, f32), anchor: Anchor, kind: RelativeDepthFitKind) -> f32 {
    let input = match kind {
        RelativeDepthFitKind::Direct => anchor.learned,
        RelativeDepthFitKind::Inverse => 1.0 / anchor.learned,
    };
    let predicted = fit.0 * input + fit.1;
    if !predicted.is_finite() || predicted <= 1.0e-6 {
        return f32::INFINITY;
    }
    (predicted - anchor.geometric_depth).abs() / anchor.geometric_depth.max(1.0e-6)
}

fn median_sorted(values: &[f32]) -> f32 {
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_camera() -> LearnedDepthCamera {
        LearnedDepthCamera {
            frame_index: 0,
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
        }
    }

    fn fixture(inverse: bool) -> (Vec<Point3>, Vec<f32>) {
        let width = 96usize;
        let height = 72usize;
        let focal = 60.0f32;
        let mut depths = vec![1.0; width * height];
        let mut points = Vec::new();
        for index in 0..20usize {
            let u = 12 + (index % 5) * 16;
            let v = 12 + (index / 5) * 14;
            let z = 1.5 + index as f32 * 0.08;
            let x = (u as f32 - width as f32 * 0.5) / focal * z;
            let y = (v as f32 - height as f32 * 0.5) / focal * z;
            points.push(Point3 {
                x,
                y,
                z,
                confidence: 1.0,
                r: 0,
                g: 0,
                b: 0,
            });
            depths[v * width + u] = if inverse { 1.0 / z } else { (z - 0.3) / 1.7 };
        }
        (points, depths)
    }

    #[test]
    fn recovers_direct_relative_depth_alignment() {
        let (points, depths) = fixture(false);
        let frame = RelativeDepthFrame {
            frame_index: 0,
            width: 96,
            height: 72,
            values: &depths,
        };
        let result = evaluate_relative_depth(&[identity_camera()], &points, 60.0, &[frame]);
        let diagnostics = &result.frames[0];
        assert_eq!(diagnostics.fit_kind, Some(RelativeDepthFitKind::Direct));
        assert!(diagnostics.calibration_usable);
        assert!(diagnostics.median_relative_error.unwrap() < 1.0e-4);
    }

    #[test]
    fn recovers_inverse_relative_depth_alignment() {
        let (points, depths) = fixture(true);
        let frame = RelativeDepthFrame {
            frame_index: 0,
            width: 96,
            height: 72,
            values: &depths,
        };
        let result = evaluate_relative_depth(&[identity_camera()], &points, 60.0, &[frame]);
        let diagnostics = &result.frames[0];
        assert_eq!(diagnostics.fit_kind, Some(RelativeDepthFitKind::Inverse));
        assert!(diagnostics.calibration_usable);
        assert!(diagnostics.median_relative_error.unwrap() < 1.0e-4);
    }

    #[test]
    fn fails_closed_when_alignment_has_too_few_anchors() {
        let points = vec![Point3 {
            x: 0.0,
            y: 0.0,
            z: 2.0,
            confidence: 1.0,
            r: 0,
            g: 0,
            b: 0,
        }];
        let depths = vec![1.0; 16 * 16];
        let frame = RelativeDepthFrame {
            frame_index: 0,
            width: 16,
            height: 16,
            values: &depths,
        };
        let result = evaluate_relative_depth(&[identity_camera()], &points, 20.0, &[frame]);
        assert!(!result.frames[0].calibration_usable);
        assert!(result.frames[0]
            .skip_reason
            .as_deref()
            .unwrap()
            .contains("fewer than"));
    }
}
