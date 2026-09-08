use nalgebra::{DMatrix, Matrix3, SMatrix, Vector3};
use std::cmp::Ordering;

const DLT_SAMPLE_SIZE: usize = 6;
const MIN_ACCEPTED_INLIERS: usize = 8;
const MIN_INLIER_RATIO: f64 = 0.55;
const MAX_REPROJECTION_ERROR_PIXELS: f64 = 4.0;
const MAX_MEDIAN_REPROJECTION_ERROR_PIXELS: f64 = 2.5;
const RANSAC_TRIALS: usize = 256;

#[derive(Clone, Copy, Debug)]
pub(super) struct PnpCorrespondence {
    pub point: Vector3<f64>,
    pub x_pixels: f64,
    pub y_pixels: f64,
}

#[derive(Clone, Debug)]
pub(super) struct PnpEstimate {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub camera_center: Vector3<f64>,
    pub inliers: usize,
    pub median_reprojection_error_pixels: f64,
}

#[derive(Clone, Debug)]
struct EvaluatedPose {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    inlier_indices: Vec<usize>,
    median_reprojection_error_pixels: f64,
}

pub(super) fn estimate_pose(
    correspondences: &[PnpCorrespondence],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<PnpEstimate> {
    if correspondences.len() < MIN_ACCEPTED_INLIERS
        || !focal_pixels.is_finite()
        || focal_pixels <= 0.0
    {
        return None;
    }

    let mut best: Option<EvaluatedPose> = None;
    for trial in 0..RANSAC_TRIALS {
        let Some(sample_indices) = deterministic_sample(trial, correspondences.len()) else {
            continue;
        };
        let sample: Vec<PnpCorrespondence> = sample_indices
            .iter()
            .map(|&index| correspondences[index])
            .collect();
        let Some((rotation, translation)) = fit_dlt(&sample, width, height, focal_pixels) else {
            continue;
        };
        let evaluated = evaluate_pose(
            correspondences,
            rotation,
            translation,
            width,
            height,
            focal_pixels,
        );
        if evaluated.inlier_indices.len() < MIN_ACCEPTED_INLIERS {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|current| is_better(&evaluated, current))
        {
            best = Some(evaluated);
        }
    }

    let best = best?;
    let inlier_correspondences: Vec<PnpCorrespondence> = best
        .inlier_indices
        .iter()
        .map(|&index| correspondences[index])
        .collect();
    let refined = fit_dlt(&inlier_correspondences, width, height, focal_pixels)
        .map(|(rotation, translation)| {
            evaluate_pose(
                correspondences,
                rotation,
                translation,
                width,
                height,
                focal_pixels,
            )
        })
        .filter(|candidate| passes_acceptance(candidate, correspondences.len()))
        .filter(|candidate| is_better(candidate, &best));
    let accepted = refined.as_ref().unwrap_or(&best);

    if !passes_acceptance(accepted, correspondences.len()) {
        return None;
    }

    Some(PnpEstimate {
        rotation: accepted.rotation,
        translation: accepted.translation,
        camera_center: -accepted.rotation.transpose() * accepted.translation,
        inliers: accepted.inlier_indices.len(),
        median_reprojection_error_pixels: accepted.median_reprojection_error_pixels,
    })
}

fn fit_dlt(
    correspondences: &[PnpCorrespondence],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    if correspondences.len() < DLT_SAMPLE_SIZE {
        return None;
    }

    let centroid = correspondences
        .iter()
        .fold(Vector3::zeros(), |sum, correspondence| sum + correspondence.point)
        / correspondences.len() as f64;
    let rms = (correspondences
        .iter()
        .map(|correspondence| (correspondence.point - centroid).norm_squared())
        .sum::<f64>()
        / correspondences.len() as f64)
        .sqrt();
    if !rms.is_finite() || rms < 1e-8 {
        return None;
    }

    let center_x = width as f64 * 0.5;
    let center_y = height as f64 * 0.5;
    let mut design = DMatrix::<f64>::zeros(correspondences.len() * 2, 12);
    for (index, correspondence) in correspondences.iter().enumerate() {
        let normalized = (correspondence.point - centroid) / rms;
        let world = [normalized.x, normalized.y, normalized.z, 1.0];
        let x = (correspondence.x_pixels - center_x) / focal_pixels;
        let y = (correspondence.y_pixels - center_y) / focal_pixels;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }

        let row_x = index * 2;
        let row_y = row_x + 1;
        for column in 0..4 {
            design[(row_x, column)] = world[column];
            design[(row_x, 8 + column)] = -x * world[column];
            design[(row_y, 4 + column)] = world[column];
            design[(row_y, 8 + column)] = -y * world[column];
        }
    }

    let svd = design.svd(false, true);
    let v_t = svd.v_t?;
    let solution = v_t.row(v_t.nrows() - 1);
    let mut normalized_projection = SMatrix::<f64, 3, 4>::zeros();
    for row in 0..3 {
        for column in 0..4 {
            normalized_projection[(row, column)] = solution[row * 4 + column];
        }
    }

    let mut world_normalization = SMatrix::<f64, 4, 4>::identity();
    world_normalization[(0, 0)] = 1.0 / rms;
    world_normalization[(1, 1)] = 1.0 / rms;
    world_normalization[(2, 2)] = 1.0 / rms;
    world_normalization[(0, 3)] = -centroid.x / rms;
    world_normalization[(1, 3)] = -centroid.y / rms;
    world_normalization[(2, 3)] = -centroid.z / rms;
    let mut projection = normalized_projection * world_normalization;

    let mut linear = projection.fixed_view::<3, 3>(0, 0).into_owned();
    if linear.determinant() < 0.0 {
        projection *= -1.0;
        linear *= -1.0;
    }
    let linear_svd = linear.svd(true, true);
    let scale = linear_svd.singular_values.iter().sum::<f64>() / 3.0;
    if !scale.is_finite() || scale < 1e-10 {
        return None;
    }

    let raw_rotation = linear / scale;
    let rotation_svd = raw_rotation.svd(true, true);
    let mut u = rotation_svd.u?;
    let v_t = rotation_svd.v_t?;
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        for row in 0..3 {
            u[(row, 2)] *= -1.0;
        }
        rotation = u * v_t;
    }
    if !rotation.iter().all(|value| value.is_finite()) {
        return None;
    }

    let translation = Vector3::new(
        projection[(0, 3)] / scale,
        projection[(1, 3)] / scale,
        projection[(2, 3)] / scale,
    );
    if !translation.iter().all(|value| value.is_finite()) {
        return None;
    }

    Some((rotation, translation))
}

fn evaluate_pose(
    correspondences: &[PnpCorrespondence],
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> EvaluatedPose {
    let center_x = width as f64 * 0.5;
    let center_y = height as f64 * 0.5;
    let mut inlier_indices = Vec::new();
    let mut inlier_errors = Vec::new();

    for (index, correspondence) in correspondences.iter().enumerate() {
        let camera_point = rotation * correspondence.point + translation;
        if !camera_point.iter().all(|value| value.is_finite()) || camera_point.z <= 1e-6 {
            continue;
        }
        let projected_x = focal_pixels * camera_point.x / camera_point.z + center_x;
        let projected_y = focal_pixels * camera_point.y / camera_point.z + center_y;
        let error = (projected_x - correspondence.x_pixels)
            .hypot(projected_y - correspondence.y_pixels);
        if error.is_finite() && error <= MAX_REPROJECTION_ERROR_PIXELS {
            inlier_indices.push(index);
            inlier_errors.push(error);
        }
    }

    EvaluatedPose {
        rotation,
        translation,
        inlier_indices,
        median_reprojection_error_pixels: median(&mut inlier_errors),
    }
}

fn passes_acceptance(candidate: &EvaluatedPose, total_correspondences: usize) -> bool {
    candidate.inlier_indices.len() >= MIN_ACCEPTED_INLIERS
        && candidate.inlier_indices.len() as f64 / total_correspondences as f64 >= MIN_INLIER_RATIO
        && candidate.median_reprojection_error_pixels <= MAX_MEDIAN_REPROJECTION_ERROR_PIXELS
}

fn is_better(candidate: &EvaluatedPose, current: &EvaluatedPose) -> bool {
    candidate.inlier_indices.len() > current.inlier_indices.len()
        || (candidate.inlier_indices.len() == current.inlier_indices.len()
            && candidate
                .median_reprojection_error_pixels
                .partial_cmp(&current.median_reprojection_error_pixels)
                .unwrap_or(Ordering::Greater)
                == Ordering::Less)
}

fn deterministic_sample(trial: usize, len: usize) -> Option<[usize; DLT_SAMPLE_SIZE]> {
    if len < DLT_SAMPLE_SIZE {
        return None;
    }
    let mut state = (trial as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    let mut sample = [0usize; DLT_SAMPLE_SIZE];
    for slot_index in 0..DLT_SAMPLE_SIZE {
        let mut attempts = 0usize;
        loop {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let index = (state.wrapping_mul(0x2545f4914f6cdd1d) % len as u64) as usize;
            if !sample[..slot_index].contains(&index) {
                sample[slot_index] = index;
                break;
            }
            attempts += 1;
            if attempts > len * 4 {
                return None;
            }
        }
    }
    Some(sample)
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn synthetic_correspondences() -> (Vec<PnpCorrespondence>, Vector3<f64>) {
        let width = 640u32;
        let height = 480u32;
        let focal = 520.0;
        let rotation = Rotation3::from_euler_angles(0.04, -0.08, 0.03).into_inner();
        let translation = Vector3::new(0.18, -0.12, 4.2);
        let camera_center = -rotation.transpose() * translation;
        let mut correspondences = Vec::new();

        for index in 0..24 {
            let point = Vector3::new(
                (index % 4) as f64 * 0.55 - 0.8,
                ((index / 4) % 3) as f64 * 0.5 - 0.5,
                (index % 5) as f64 * 0.28 + (index / 12) as f64 * 0.35,
            );
            let camera_point = rotation * point + translation;
            let mut x = focal * camera_point.x / camera_point.z + width as f64 * 0.5;
            let mut y = focal * camera_point.y / camera_point.z + height as f64 * 0.5;
            x += (index as f64 * 0.37).sin() * 0.18;
            y += (index as f64 * 0.29).cos() * 0.18;
            if matches!(index, 3 | 11 | 19) {
                x += 32.0;
                y -= 26.0;
            }
            correspondences.push(PnpCorrespondence {
                point,
                x_pixels: x,
                y_pixels: y,
            });
        }

        (correspondences, camera_center)
    }

    #[test]
    fn robust_pnp_recovers_camera_with_outliers() {
        let (correspondences, expected_center) = synthetic_correspondences();
        let estimate = estimate_pose(&correspondences, 640, 480, 520.0)
            .expect("synthetic camera should register");

        assert!(estimate.inliers >= 20, "only {} PnP inliers", estimate.inliers);
        assert!(estimate.median_reprojection_error_pixels < 0.7);
        assert!((estimate.camera_center - expected_center).norm() < 0.08);
        assert!((estimate.rotation.determinant() - 1.0).abs() < 1e-6);
        assert!(estimate.translation.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn pnp_rejects_underconstrained_inputs() {
        let (mut correspondences, _) = synthetic_correspondences();
        correspondences.truncate(7);
        assert!(estimate_pose(&correspondences, 640, 480, 520.0).is_none());
    }
}
