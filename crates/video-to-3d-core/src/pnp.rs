use nalgebra::{DMatrix, Matrix3, SMatrix, SymmetricEigen, Vector3};
use std::cmp::Ordering;

const DLT_SAMPLE_SIZE: usize = 6;
const MIN_ACCEPTED_INLIERS: usize = 8;
const MIN_INLIER_RATIO: f64 = 0.55;
const MAX_REPROJECTION_ERROR_PIXELS: f64 = 4.0;
const MAX_MEDIAN_REPROJECTION_ERROR_PIXELS: f64 = 2.5;
const PLANAR_VARIANCE_RATIO: f64 = 0.01;
const COLLINEAR_VARIANCE_RATIO: f64 = 1e-4;
const DESIGN_RANK_RATIO: f64 = 1e-8;
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

#[derive(Clone, Copy, Debug)]
struct PlaneBasis {
    centroid: Vector3<f64>,
    axis_u: Vector3<f64>,
    axis_v: Vector3<f64>,
    normal: Vector3<f64>,
}

enum LandmarkGeometry {
    Spatial,
    Planar(PlaneBasis),
    Degenerate,
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
        let Some((rotation, translation)) = fit_pose(&sample, width, height, focal_pixels) else {
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
    let refined = fit_pose(&inlier_correspondences, width, height, focal_pixels)
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

fn fit_pose(
    correspondences: &[PnpCorrespondence],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    match classify_landmarks(correspondences) {
        LandmarkGeometry::Spatial => fit_spatial_dlt(correspondences, width, height, focal_pixels),
        LandmarkGeometry::Planar(plane) => {
            fit_planar_homography(correspondences, plane, width, height, focal_pixels)
        }
        LandmarkGeometry::Degenerate => None,
    }
}

fn classify_landmarks(correspondences: &[PnpCorrespondence]) -> LandmarkGeometry {
    if correspondences.len() < DLT_SAMPLE_SIZE {
        return LandmarkGeometry::Degenerate;
    }

    let centroid = correspondences
        .iter()
        .fold(Vector3::zeros(), |sum, correspondence| sum + correspondence.point)
        / correspondences.len() as f64;
    let covariance = correspondences.iter().fold(Matrix3::zeros(), |sum, correspondence| {
        let delta = correspondence.point - centroid;
        sum + delta * delta.transpose()
    }) / correspondences.len() as f64;
    let eigen = SymmetricEigen::new(covariance);
    let mut indices = [0usize, 1, 2];
    indices.sort_by(|left, right| eigen.eigenvalues[*left].total_cmp(&eigen.eigenvalues[*right]));

    let smallest = eigen.eigenvalues[indices[0]].max(0.0);
    let middle = eigen.eigenvalues[indices[1]].max(0.0);
    let largest = eigen.eigenvalues[indices[2]].max(0.0);
    if !largest.is_finite() || largest < 1e-10 || middle / largest < COLLINEAR_VARIANCE_RATIO {
        return LandmarkGeometry::Degenerate;
    }
    if smallest / largest > PLANAR_VARIANCE_RATIO {
        return LandmarkGeometry::Spatial;
    }

    let mut axis_u = eigen.eigenvectors.column(indices[2]).into_owned();
    let mut normal = eigen.eigenvectors.column(indices[0]).into_owned();
    canonicalize_direction(&mut axis_u);
    canonicalize_direction(&mut normal);
    axis_u = axis_u.normalize();
    normal = normal.normalize();
    let axis_v = normal.cross(&axis_u).normalize();
    LandmarkGeometry::Planar(PlaneBasis {
        centroid,
        axis_u,
        axis_v,
        normal,
    })
}

fn canonicalize_direction(direction: &mut Vector3<f64>) {
    let mut dominant = 0usize;
    for index in 1..3 {
        if direction[index].abs() > direction[dominant].abs() {
            dominant = index;
        }
    }
    if direction[dominant] < 0.0 {
        *direction *= -1.0;
    }
}

fn fit_spatial_dlt(
    correspondences: &[PnpCorrespondence],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
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
    if has_extra_null_direction(&svd.singular_values) {
        return None;
    }
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

    let rotation = project_to_rotation(linear / scale)?;
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

fn fit_planar_homography(
    correspondences: &[PnpCorrespondence],
    plane: PlaneBasis,
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let plane_coordinates: Vec<(f64, f64)> = correspondences
        .iter()
        .map(|correspondence| {
            let delta = correspondence.point - plane.centroid;
            (delta.dot(&plane.axis_u), delta.dot(&plane.axis_v))
        })
        .collect();
    let rms = (plane_coordinates
        .iter()
        .map(|(u, v)| u * u + v * v)
        .sum::<f64>()
        / plane_coordinates.len() as f64)
        .sqrt();
    if !rms.is_finite() || rms < 1e-8 {
        return None;
    }

    let center_x = width as f64 * 0.5;
    let center_y = height as f64 * 0.5;
    let mut design = DMatrix::<f64>::zeros(correspondences.len() * 2, 9);
    for (index, (correspondence, &(u, v))) in correspondences
        .iter()
        .zip(&plane_coordinates)
        .enumerate()
    {
        let u = u / rms;
        let v = v / rms;
        let x = (correspondence.x_pixels - center_x) / focal_pixels;
        let y = (correspondence.y_pixels - center_y) / focal_pixels;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }

        let row_x = index * 2;
        let row_y = row_x + 1;
        design[(row_x, 0)] = u;
        design[(row_x, 1)] = v;
        design[(row_x, 2)] = 1.0;
        design[(row_x, 6)] = -x * u;
        design[(row_x, 7)] = -x * v;
        design[(row_x, 8)] = -x;
        design[(row_y, 3)] = u;
        design[(row_y, 4)] = v;
        design[(row_y, 5)] = 1.0;
        design[(row_y, 6)] = -y * u;
        design[(row_y, 7)] = -y * v;
        design[(row_y, 8)] = -y;
    }

    let svd = design.svd(false, true);
    if has_extra_null_direction(&svd.singular_values) {
        return None;
    }
    let v_t = svd.v_t?;
    let solution = v_t.row(v_t.nrows() - 1);
    let mut homography = Matrix3::zeros();
    for row in 0..3 {
        for column in 0..3 {
            homography[(row, column)] = solution[row * 3 + column];
        }
    }
    homography *= Matrix3::new(
        1.0 / rms,
        0.0,
        0.0,
        0.0,
        1.0 / rms,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let h1 = homography.column(0).into_owned();
    let h2 = homography.column(1).into_owned();
    let h3 = homography.column(2).into_owned();
    let scale = (h1.norm() + h2.norm()) * 0.5;
    if !scale.is_finite() || scale < 1e-10 {
        return None;
    }

    let world_basis = Matrix3::from_columns(&[plane.axis_u, plane.axis_v, plane.normal]);
    let mut best: Option<EvaluatedPose> = None;
    for sign in [1.0, -1.0] {
        let camera_u = h1 * (sign / scale);
        let camera_v = h2 * (sign / scale);
        let camera_normal = camera_u.cross(&camera_v);
        if camera_normal.norm_squared() < 1e-12 {
            continue;
        }
        let camera_basis = project_to_rotation(Matrix3::from_columns(&[
            camera_u,
            camera_v,
            camera_normal.normalize(),
        ]))?;
        let rotation = project_to_rotation(camera_basis * world_basis.transpose())?;
        let translation = h3 * (sign / scale) - rotation * plane.centroid;
        let evaluated = evaluate_pose(
            correspondences,
            rotation,
            translation,
            width,
            height,
            focal_pixels,
        );
        if best
            .as_ref()
            .is_none_or(|current| is_better(&evaluated, current))
        {
            best = Some(evaluated);
        }
    }

    best.map(|candidate| (candidate.rotation, candidate.translation))
}

fn project_to_rotation(raw: Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = raw.svd(true, true);
    let mut u = svd.u?;
    let v_t = svd.v_t?;
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        for row in 0..3 {
            u[(row, 2)] *= -1.0;
        }
        rotation = u * v_t;
    }
    if rotation.iter().all(|value| value.is_finite()) {
        Some(rotation)
    } else {
        None
    }
}

fn has_extra_null_direction(singular_values: &nalgebra::DVector<f64>) -> bool {
    if singular_values.len() < 2 {
        return true;
    }
    let largest = singular_values[0].abs();
    let second_smallest = singular_values[singular_values.len() - 2].abs();
    !largest.is_finite()
        || largest < 1e-12
        || second_smallest / largest < DESIGN_RANK_RATIO
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

    fn planar_correspondences() -> (Vec<PnpCorrespondence>, Vector3<f64>) {
        let width = 640u32;
        let height = 480u32;
        let focal = 520.0;
        let rotation = Rotation3::from_euler_angles(-0.07, 0.13, -0.04).into_inner();
        let translation = Vector3::new(-0.15, 0.09, 4.6);
        let camera_center = -rotation.transpose() * translation;
        let mut correspondences = Vec::new();

        for index in 0..20 {
            let point = Vector3::new(
                (index % 5) as f64 * 0.42 - 0.84,
                (index / 5) as f64 * 0.38 - 0.57,
                ((index % 3) as f64 - 1.0) * 0.0002,
            );
            let camera_point = rotation * point + translation;
            let mut x = focal * camera_point.x / camera_point.z + width as f64 * 0.5;
            let mut y = focal * camera_point.y / camera_point.z + height as f64 * 0.5;
            x += (index as f64 * 0.23).sin() * 0.12;
            y += (index as f64 * 0.31).cos() * 0.12;
            if matches!(index, 4 | 14) {
                x -= 28.0;
                y += 24.0;
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
    fn robust_pnp_recovers_nearly_planar_camera_with_outliers() {
        let (correspondences, expected_center) = planar_correspondences();
        let estimate = estimate_pose(&correspondences, 640, 480, 520.0)
            .expect("nearly planar landmarks should use the homography pose path");

        assert!(estimate.inliers >= 17, "only {} planar PnP inliers", estimate.inliers);
        assert!(estimate.median_reprojection_error_pixels < 0.7);
        assert!((estimate.camera_center - expected_center).norm() < 0.08);
        assert!((estimate.rotation.determinant() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pnp_rejects_underconstrained_inputs() {
        let (mut correspondences, _) = synthetic_correspondences();
        correspondences.truncate(7);
        assert!(estimate_pose(&correspondences, 640, 480, 520.0).is_none());
    }
}
