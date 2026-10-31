//! Seed-pair candidates: wide-span track composition, sub-pixel refinement, and the
//! evidence and statistics recorded for a seed.

use super::matching::{compensate_global_motion, median};
use super::types::{
    CalibratedPairStats, CameraPose, Feature, FeatureMatch, SeedCandidateStats, SeedGate,
};
use crate::two_view;
use std::collections::HashMap;

/// Widest seed pair, in sampling intervals, tried when no adjacent pair of a segment seeds.
pub(super) const MAX_SEED_SPAN_INTERVALS: usize = 6;

/// Matches from the first to the last frame of consecutive adjacent links, following each
/// feature through every link. Only the end frames' detected positions are used, so the
/// composed correspondence carries no accumulated interpolation; its descriptor distance is
/// the worst link's.
pub(super) fn compose_track_matches(links: &[Vec<FeatureMatch>]) -> Vec<FeatureMatch> {
    let Some((first, rest)) = links.split_first() else {
        return Vec::new();
    };
    let mut composed: Vec<FeatureMatch> = first.clone();
    for link in rest {
        let next: HashMap<usize, &FeatureMatch> = link
            .iter()
            .map(|feature_match| (feature_match.a, feature_match))
            .collect();
        composed = composed
            .into_iter()
            .filter_map(|feature_match| {
                next.get(&feature_match.b).map(|step| FeatureMatch {
                    a: feature_match.a,
                    b: step.b,
                    distance: feature_match.distance.max(step.distance),
                })
            })
            .collect();
    }
    composed.sort_by_key(|feature_match| feature_match.a);
    composed
}

/// Evaluate a non-adjacent seed pair. Its matches are refined to sub-pixel positions, since
/// whole-pixel rounding is close to the rotation-only gate at this parallax. The pair must
/// carry measurable translation: a least-median pure-rotation fit over every refined match
/// must leave more than the rotation-only gate, so a pure rotation (whose wide pairs fit
/// it to about a tenth of a pixel) is rejected before an essential matrix is tried. Then
/// the shared two-view gates apply unchanged.
#[cfg(test)]
pub(super) fn wide_seed_candidate(
    [source, target]: [&[Feature]; 2],
    luma: [&[u8]; 2],
    matches: &[FeatureMatch],
    seed_frames: [usize; 2],
    width: u32,
    height: u32,
    focal: f32,
) -> (SeedCandidateStats, Option<two_view::TwoViewEstimate>) {
    wide_seed_candidate_with_control(
        [source, target],
        luma,
        matches,
        seed_frames,
        width,
        height,
        focal,
        &mut crate::ReconstructionControl::new(&mut || false),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn wide_seed_candidate_with_control(
    [source, target]: [&[Feature]; 2],
    luma: [&[u8]; 2],
    matches: &[FeatureMatch],
    seed_frames: [usize; 2],
    width: u32,
    height: u32,
    focal: f32,
    control: &mut crate::ReconstructionControl<'_>,
) -> (SeedCandidateStats, Option<two_view::TwoViewEstimate>) {
    let mut motion: Vec<f32> = matches
        .iter()
        .map(|m| {
            let (a, b) = (&source[m.a], &target[m.b]);
            (b.x as f32 - a.x as f32).hypot(b.y as f32 - a.y as f32)
        })
        .collect();
    let median_motion = median(&mut motion);
    let (_, _, mut residuals) = compensate_global_motion(source, target, matches, width, height);
    let median_parallax_residual = median(&mut residuals);
    let mut seed = SeedCandidateStats {
        from_frame: seed_frames[0],
        to_frame: seed_frames[1],
        matches: matches.len(),
        median_motion,
        median_parallax_residual,
        rejected_gate: None,
        selected: false,
        inliers: None,
        inlier_ratio: None,
        rotation_only_residual_pixels: None,
        required_points: None,
        triangulated_points: None,
        behind_camera: None,
        failed_triangulations: None,
        high_reprojection: None,
        median_reprojection_error_pixels: None,
        median_triangulation_angle_degrees: None,
    };
    if matches.len() < 6 || median_motion < 1.4 {
        seed.rejected_gate = Some(SeedGate::Parallax);
        return (seed, None);
    }
    let points: Vec<two_view::PixelCorrespondence> = matches
        .iter()
        .filter_map(|m| {
            let (a, b) = (&source[m.a], &target[m.b]);
            let refined = refine_subpixel(luma, width, height, [a.x, a.y], [b.x, b.y])?;
            Some(two_view::PixelCorrespondence {
                source_feature_index: m.a,
                descriptor_distance: m.distance,
                source: [a.x as f64, a.y as f64],
                target: refined,
            })
        })
        .collect();
    // The candidate reports the correspondences actually evaluated.
    seed.matches = points.len();
    if points.len() < two_view::MIN_SEED_INLIERS {
        // Too few matches survived sub-pixel refinement (e.g. near the border)
        // to fit anything: a correspondence limit, not a missing baseline.
        seed.rejected_gate = Some(SeedGate::Matches);
        return (seed, None);
    }
    let rotation_residual = two_view::least_median_rotation_residual_pixels_with_control(
        &points,
        width,
        height,
        focal as f64,
        control,
    );
    if rotation_residual
        .is_none_or(|residual| residual <= two_view::MIN_SEED_ROTATION_RESIDUAL_PIXELS)
    {
        seed.rotation_only_residual_pixels = rotation_residual.map(|value| value as f32);
        seed.rejected_gate = Some(SeedGate::Baseline);
        return (seed, None);
    }
    let outcome = two_view::evaluate_two_view_points_with_control(
        &points,
        width,
        height,
        focal as f64,
        control,
    );
    record_two_view_evidence(&mut seed, &outcome.evidence);
    // A wide pair reports the least-median fit it was gated on, unless the
    // shared baseline gate rejected it: then the residual that decided stays.
    if outcome.result.as_ref().err() != Some(&two_view::TwoViewRejection::Baseline) {
        seed.rotation_only_residual_pixels = rotation_residual.map(|value| value as f32);
    }
    match outcome.result {
        Ok(estimate) => (seed, Some(estimate)),
        Err(rejection) => {
            seed.rejected_gate = Some(SeedGate::from_two_view(rejection));
            (seed, None)
        }
    }
}

/// Half-size of the patch compared when refining a match to sub-pixel accuracy.
const SUBPIXEL_PATCH_RADIUS: i64 = 3;

/// Target position of a match refined to sub-pixel accuracy: the integer offset within
/// one pixel minimizing the patch SSD, then a separable parabola through its neighbours.
/// `None` near the border or when the minimum is not a strict interior minimum.
pub(super) fn refine_subpixel(
    [source, target]: [&[u8]; 2],
    width: u32,
    height: u32,
    a: [u32; 2],
    b: [u32; 2],
) -> Option<[f64; 2]> {
    let (w, h) = (i64::from(width), i64::from(height));
    let reach = SUBPIXEL_PATCH_RADIUS + 2;
    let inside = |[x, y]: [u32; 2]| {
        let (x, y) = (i64::from(x), i64::from(y));
        x >= reach && y >= reach && x < w - reach && y < h - reach
    };
    if !inside(a) || !inside(b) {
        return None;
    }
    let ssd = |dx: i64, dy: i64| -> f64 {
        let mut total = 0.0;
        for oy in -SUBPIXEL_PATCH_RADIUS..=SUBPIXEL_PATCH_RADIUS {
            for ox in -SUBPIXEL_PATCH_RADIUS..=SUBPIXEL_PATCH_RADIUS {
                let s = source[((i64::from(a[1]) + oy) * w + i64::from(a[0]) + ox) as usize];
                let t =
                    target[((i64::from(b[1]) + oy + dy) * w + i64::from(b[0]) + ox + dx) as usize];
                let d = f64::from(s) - f64::from(t);
                total += d * d;
            }
        }
        total
    };
    let mut best = (0_i64, 0_i64, ssd(0, 0));
    for dy in -1..=1 {
        for dx in -1..=1 {
            let value = ssd(dx, dy);
            if value < best.2 {
                best = (dx, dy, value);
            }
        }
    }
    let (dx, dy, center) = best;
    let vertex = |minus: f64, plus: f64| -> Option<f64> {
        let curvature = minus + plus - 2.0 * center;
        (curvature > 0.0).then(|| (0.5 * (minus - plus) / curvature).clamp(-0.5, 0.5))
    };
    let offset_x = vertex(ssd(dx - 1, dy), ssd(dx + 1, dy))?;
    let offset_y = vertex(ssd(dx, dy - 1), ssd(dx, dy + 1))?;
    Some([
        (i64::from(b[0]) + dx) as f64 + offset_x,
        (i64::from(b[1]) + dy) as f64 + offset_y,
    ])
}

pub(super) fn record_two_view_evidence(
    seed: &mut SeedCandidateStats,
    evidence: &two_view::TwoViewEvidence,
) {
    if evidence.matches < two_view::MIN_SEED_INLIERS {
        return;
    }
    seed.inliers = Some(evidence.inliers);
    seed.inlier_ratio = Some(evidence.inliers as f32 / evidence.matches as f32);
    seed.rotation_only_residual_pixels = evidence
        .rotation_only_residual_pixels
        .map(|value| value as f32);
    if let Some(pose) = evidence.pose {
        seed.required_points = Some(pose.required_points);
        seed.triangulated_points = Some(pose.accepted_points);
        seed.behind_camera = Some(pose.behind_camera);
        seed.failed_triangulations = Some(pose.failed_triangulation);
        seed.high_reprojection = Some(pose.high_reprojection);
        seed.median_reprojection_error_pixels = pose
            .median_reprojection_error_pixels
            .map(|value| value as f32);
        seed.median_triangulation_angle_degrees = pose
            .median_triangulation_angle_degrees
            .map(|value| value as f32);
    }
}

/// The best seed candidate among the pairs `include` accepts, using the same ordering
/// and tie-break (earliest pair wins) as a single sequential scan.
pub(super) fn best_seed_index(
    candidates: &[([usize; 2], two_view::TwoViewEstimate)],
    include: impl Fn(usize) -> bool,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, (seed_frames, estimate)) in candidates.iter().enumerate() {
        if include(seed_frames[0])
            && best.is_none_or(|current| estimate.is_better_than(&candidates[current].1))
        {
            best = Some(index);
        }
    }
    best
}

pub(super) fn seed_pair_cameras(
    seed_frames: [usize; 2],
    estimate: &two_view::TwoViewEstimate,
) -> Vec<CameraPose> {
    vec![
        CameraPose {
            frame_index: seed_frames[0],
            x: 0.0,
            y: 0.0,
            z: 0.0,
            matched_features: 0,
        },
        CameraPose {
            frame_index: seed_frames[1],
            x: estimate.camera_center.x as f32,
            y: estimate.camera_center.y as f32,
            z: estimate.camera_center.z as f32,
            matched_features: estimate.inliers,
        },
    ]
}

pub(super) fn calibrated_pair_stats(
    seed_frames: [usize; 2],
    estimate: &two_view::TwoViewEstimate,
    focal: f32,
) -> CalibratedPairStats {
    CalibratedPairStats {
        from_frame: seed_frames[0],
        to_frame: seed_frames[1],
        matches: estimate.matches,
        inliers: estimate.inliers,
        inlier_ratio: estimate.inliers as f32 / estimate.matches as f32,
        focal_pixels: focal,
        median_sampson_error_pixels: estimate.median_sampson_error_pixels as f32,
        median_reprojection_error_pixels: estimate.median_reprojection_error_pixels as f32,
        median_triangulation_angle_degrees: estimate.median_triangulation_angle_degrees as f32,
        relative_rotation: [
            estimate.rotation[(0, 0)] as f32,
            estimate.rotation[(0, 1)] as f32,
            estimate.rotation[(0, 2)] as f32,
            estimate.rotation[(1, 0)] as f32,
            estimate.rotation[(1, 1)] as f32,
            estimate.rotation[(1, 2)] as f32,
            estimate.rotation[(2, 0)] as f32,
            estimate.rotation[(2, 1)] as f32,
            estimate.rotation[(2, 2)] as f32,
        ],
        translation_direction: [
            estimate.translation.x as f32,
            estimate.translation.y as f32,
            estimate.translation.z as f32,
        ],
    }
}
