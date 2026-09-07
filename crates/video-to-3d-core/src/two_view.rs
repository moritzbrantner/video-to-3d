use super::{Feature, FeatureMatch};
use nalgebra::{DMatrix, Matrix3, Matrix4, Vector3};
use std::cmp::Ordering;

const RANSAC_ITERATIONS: usize = 96;
const SAMPSON_THRESHOLD_PIXELS: f64 = 1.75;
const MIN_INLIER_RATIO: f64 = 0.35;
const MAX_ROTATION_ONLY_RESIDUAL_PIXELS: f64 = 1.25;
const MIN_TRIANGULATION_ANGLE_DEGREES: f64 = 0.5;
const MAX_REPROJECTION_ERROR_PIXELS: f64 = 4.0;

#[derive(Clone, Debug)]
pub(super) struct TriangulatedPoint {
    pub source_feature_index: usize,
    pub descriptor_distance: f32,
    pub position: Vector3<f64>,
    pub reprojection_error_pixels: f64,
    pub triangulation_angle_degrees: f64,
}

#[derive(Clone, Debug)]
pub(super) struct TwoViewEstimate {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub camera_center: Vector3<f64>,
    pub matches: usize,
    pub inliers: usize,
    pub median_sampson_error_pixels: f64,
    pub median_reprojection_error_pixels: f64,
    pub median_triangulation_angle_degrees: f64,
    pub points: Vec<TriangulatedPoint>,
}

impl TwoViewEstimate {
    pub(super) fn is_better_than(&self, other: &Self) -> bool {
        self.inliers > other.inliers
            || (self.inliers == other.inliers
                && self
                    .median_reprojection_error_pixels
                    .total_cmp(&other.median_reprojection_error_pixels)
                    == Ordering::Less)
    }
}

#[derive(Clone, Debug)]
struct Correspondence {
    source_feature_index: usize,
    descriptor_distance: f32,
    x1: Vector3<f64>,
    x2: Vector3<f64>,
}

#[derive(Clone, Debug)]
struct PoseCandidate {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    camera_center: Vector3<f64>,
    points: Vec<TriangulatedPoint>,
    median_reprojection_error_pixels: f64,
    median_triangulation_angle_degrees: f64,
}

pub(super) fn estimate_two_view(
    source_features: &[Feature],
    target_features: &[Feature],
    matches: &[FeatureMatch],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<TwoViewEstimate> {
    if matches.len() < 8 || focal_pixels <= 0.0 {
        return None;
    }

    let center_x = width as f64 * 0.5;
    let center_y = height as f64 * 0.5;
    let correspondences: Vec<Correspondence> = matches
        .iter()
        .map(|feature_match| {
            let source = &source_features[feature_match.a];
            let target = &target_features[feature_match.b];
            Correspondence {
                source_feature_index: feature_match.a,
                descriptor_distance: feature_match.distance,
                x1: Vector3::new(
                    (source.x as f64 - center_x) / focal_pixels,
                    (source.y as f64 - center_y) / focal_pixels,
                    1.0,
                ),
                x2: Vector3::new(
                    (target.x as f64 - center_x) / focal_pixels,
                    (target.y as f64 - center_y) / focal_pixels,
                    1.0,
                ),
            }
        })
        .collect();

    let threshold = (SAMPSON_THRESHOLD_PIXELS / focal_pixels).powi(2);
    let mut best_inliers = Vec::new();
    let mut best_median_error = f64::INFINITY;

    for iteration in 0..RANSAC_ITERATIONS {
        let sample = deterministic_sample(correspondences.len(), iteration);
        let Some(essential) = estimate_essential(&correspondences, &sample) else {
            continue;
        };
        let mut inliers = Vec::new();
        let mut errors = Vec::new();
        for (index, correspondence) in correspondences.iter().enumerate() {
            let error = sampson_error(&essential, correspondence);
            if error.is_finite() && error <= threshold {
                inliers.push(index);
                errors.push(error);
            }
        }

        let median_error = median_f64(&mut errors);
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && median_error < best_median_error)
        {
            best_inliers = inliers;
            best_median_error = median_error;
        }
    }

    let inlier_ratio = best_inliers.len() as f64 / correspondences.len() as f64;
    if best_inliers.len() < 8 || inlier_ratio < MIN_INLIER_RATIO {
        return None;
    }

    if rotation_only_residual_pixels(&correspondences, &best_inliers, focal_pixels)?
        <= MAX_ROTATION_ONLY_RESIDUAL_PIXELS
    {
        return None;
    }

    let essential = estimate_essential(&correspondences, &best_inliers)?;
    let mut sampson_errors_pixels: Vec<f64> = best_inliers
        .iter()
        .map(|&index| sampson_error(&essential, &correspondences[index]).sqrt() * focal_pixels)
        .filter(|error| error.is_finite())
        .collect();
    let median_sampson_error_pixels = median_f64(&mut sampson_errors_pixels);

    let pose = recover_pose(
        &essential,
        &correspondences,
        &best_inliers,
        focal_pixels,
    )?;

    Some(TwoViewEstimate {
        rotation: pose.rotation,
        translation: pose.translation,
        camera_center: pose.camera_center,
        matches: matches.len(),
        inliers: best_inliers.len(),
        median_sampson_error_pixels,
        median_reprojection_error_pixels: pose.median_reprojection_error_pixels,
        median_triangulation_angle_degrees: pose.median_triangulation_angle_degrees,
        points: pose.points,
    })
}

fn deterministic_sample(len: usize, iteration: usize) -> Vec<usize> {
    if len == 8 {
        return (0..8).collect();
    }

    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (iteration as u64 + 1);
    let mut sample = Vec::with_capacity(8);
    while sample.len() < 8 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let index = (state % len as u64) as usize;
        if !sample.contains(&index) {
            sample.push(index);
        }
    }
    sample
}

fn estimate_essential(
    correspondences: &[Correspondence],
    indices: &[usize],
) -> Option<Matrix3<f64>> {
    if indices.len() < 8 {
        return None;
    }

    let mut design = DMatrix::<f64>::zeros(indices.len(), 9);
    for (row, &index) in indices.iter().enumerate() {
        let correspondence = &correspondences[index];
        let x1 = correspondence.x1.x;
        let y1 = correspondence.x1.y;
        let x2 = correspondence.x2.x;
        let y2 = correspondence.x2.y;
        design[(row, 0)] = x2 * x1;
        design[(row, 1)] = x2 * y1;
        design[(row, 2)] = x2;
        design[(row, 3)] = y2 * x1;
        design[(row, 4)] = y2 * y1;
        design[(row, 5)] = y2;
        design[(row, 6)] = x1;
        design[(row, 7)] = y1;
        design[(row, 8)] = 1.0;
    }

    let svd = design.svd(false, true);
    let v_t = svd.v_t?;
    let last_row = v_t.row(v_t.nrows() - 1);
    let raw = Matrix3::from_row_slice(&[
        last_row[0],
        last_row[1],
        last_row[2],
        last_row[3],
        last_row[4],
        last_row[5],
        last_row[6],
        last_row[7],
        last_row[8],
    ]);
    enforce_essential_constraints(raw)
}

fn enforce_essential_constraints(raw: Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = raw.svd(true, true);
    let mut u = svd.u?;
    let mut v_t = svd.v_t?;
    make_proper_basis(&mut u, &mut v_t);
    let sigma = (svd.singular_values[0] + svd.singular_values[1]) * 0.5;
    if !sigma.is_finite() || sigma <= f64::EPSILON {
        return None;
    }
    let diagonal = Matrix3::from_diagonal(&Vector3::new(sigma, sigma, 0.0));
    Some(u * diagonal * v_t)
}

fn make_proper_basis(u: &mut Matrix3<f64>, v_t: &mut Matrix3<f64>) {
    if u.determinant() < 0.0 {
        for row in 0..3 {
            u[(row, 2)] = -u[(row, 2)];
        }
    }
    if v_t.determinant() < 0.0 {
        for column in 0..3 {
            v_t[(2, column)] = -v_t[(2, column)];
        }
    }
}

fn sampson_error(essential: &Matrix3<f64>, correspondence: &Correspondence) -> f64 {
    let ex1 = essential * correspondence.x1;
    let etx2 = essential.transpose() * correspondence.x2;
    let numerator = correspondence.x2.dot(&ex1);
    let denominator =
        ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y;
    if denominator <= 1.0e-12 {
        return f64::INFINITY;
    }
    numerator * numerator / denominator
}

fn rotation_only_residual_pixels(
    correspondences: &[Correspondence],
    inliers: &[usize],
    focal_pixels: f64,
) -> Option<f64> {
    let mut covariance = Matrix3::<f64>::zeros();
    for &index in inliers {
        let source = correspondences[index].x1.normalize();
        let target = correspondences[index].x2.normalize();
        covariance += target * source.transpose();
    }

    let svd = covariance.svd(true, true);
    let mut u = svd.u?;
    let v_t = svd.v_t?;
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        for row in 0..3 {
            u[(row, 2)] = -u[(row, 2)];
        }
        rotation = u * v_t;
    }

    let mut residuals: Vec<f64> = inliers
        .iter()
        .map(|&index| {
            let source = correspondences[index].x1.normalize();
            let target = correspondences[index].x2.normalize();
            let predicted = rotation * source;
            predicted.dot(&target).clamp(-1.0, 1.0).acos() * focal_pixels
        })
        .filter(|residual| residual.is_finite())
        .collect();
    Some(median_f64(&mut residuals))
}

fn recover_pose(
    essential: &Matrix3<f64>,
    correspondences: &[Correspondence],
    inliers: &[usize],
    focal_pixels: f64,
) -> Option<PoseCandidate> {
    let svd = essential.svd(true, true);
    let mut u = svd.u?;
    let mut v_t = svd.v_t?;
    make_proper_basis(&mut u, &mut v_t);

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let rotations = [
        proper_rotation(u * w * v_t),
        proper_rotation(u * w.transpose() * v_t),
    ];
    let translation = Vector3::new(u[(0, 2)], u[(1, 2)], u[(2, 2)]).normalize();

    let mut best: Option<PoseCandidate> = None;
    for rotation in rotations {
        for direction in [1.0, -1.0] {
            let candidate = evaluate_pose(
                rotation,
                translation * direction,
                correspondences,
                inliers,
                focal_pixels,
            );
            if let Some(candidate) = candidate {
                let replace = best.as_ref().is_none_or(|current| {
                    candidate.points.len() > current.points.len()
                        || (candidate.points.len() == current.points.len()
                            && candidate.median_reprojection_error_pixels
                                < current.median_reprojection_error_pixels)
                });
                if replace {
                    best = Some(candidate);
                }
            }
        }
    }

    let best = best?;
    let required_cheirality = (inliers.len() * 3 / 5).max(8);
    if best.points.len() < required_cheirality
        || best.median_triangulation_angle_degrees < MIN_TRIANGULATION_ANGLE_DEGREES
    {
        return None;
    }
    Some(best)
}

fn proper_rotation(rotation: Matrix3<f64>) -> Matrix3<f64> {
    if rotation.determinant() < 0.0 {
        -rotation
    } else {
        rotation
    }
}

fn evaluate_pose(
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    correspondences: &[Correspondence],
    inliers: &[usize],
    focal_pixels: f64,
) -> Option<PoseCandidate> {
    let camera_center = -rotation.transpose() * translation;
    let mut points = Vec::new();
    let mut reprojection_errors = Vec::new();
    let mut triangulation_angles = Vec::new();

    for &index in inliers {
        let correspondence = &correspondences[index];
        let Some(position) = triangulate(
            &correspondence.x1,
            &correspondence.x2,
            &rotation,
            &translation,
        ) else {
            continue;
        };
        let camera_two_point = rotation * position + translation;
        if position.z <= 1.0e-6 || camera_two_point.z <= 1.0e-6 {
            continue;
        }

        let reprojection_error_pixels = reprojection_error(
            &position,
            &camera_two_point,
            correspondence,
            focal_pixels,
        );
        if !reprojection_error_pixels.is_finite()
            || reprojection_error_pixels > MAX_REPROJECTION_ERROR_PIXELS
        {
            continue;
        }

        let triangulation_angle_degrees = triangulation_angle(&position, &camera_center);
        if !triangulation_angle_degrees.is_finite() {
            continue;
        }

        reprojection_errors.push(reprojection_error_pixels);
        triangulation_angles.push(triangulation_angle_degrees);
        points.push(TriangulatedPoint {
            source_feature_index: correspondence.source_feature_index,
            descriptor_distance: correspondence.descriptor_distance,
            position,
            reprojection_error_pixels,
            triangulation_angle_degrees,
        });
    }

    if points.is_empty() {
        return None;
    }

    Some(PoseCandidate {
        rotation,
        translation,
        camera_center,
        points,
        median_reprojection_error_pixels: median_f64(&mut reprojection_errors),
        median_triangulation_angle_degrees: median_f64(&mut triangulation_angles),
    })
}

fn triangulate(
    x1: &Vector3<f64>,
    x2: &Vector3<f64>,
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let mut system = Matrix4::<f64>::zeros();
    let p1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let p2 = [
        [
            rotation[(0, 0)],
            rotation[(0, 1)],
            rotation[(0, 2)],
            translation.x,
        ],
        [
            rotation[(1, 0)],
            rotation[(1, 1)],
            rotation[(1, 2)],
            translation.y,
        ],
        [
            rotation[(2, 0)],
            rotation[(2, 1)],
            rotation[(2, 2)],
            translation.z,
        ],
    ];

    for column in 0..4 {
        system[(0, column)] = x1.x * p1[2][column] - p1[0][column];
        system[(1, column)] = x1.y * p1[2][column] - p1[1][column];
        system[(2, column)] = x2.x * p2[2][column] - p2[0][column];
        system[(3, column)] = x2.y * p2[2][column] - p2[1][column];
    }

    let svd = system.svd(false, true);
    let v_t = svd.v_t?;
    let homogeneous = v_t.row(3);
    let w = homogeneous[3];
    if !w.is_finite() || w.abs() <= 1.0e-12 {
        return None;
    }
    let point = Vector3::new(
        homogeneous[0] / w,
        homogeneous[1] / w,
        homogeneous[2] / w,
    );
    point
        .iter()
        .all(|value| value.is_finite())
        .then_some(point)
}

fn reprojection_error(
    point_one: &Vector3<f64>,
    point_two: &Vector3<f64>,
    correspondence: &Correspondence,
    focal_pixels: f64,
) -> f64 {
    let projected_one = Vector3::new(
        point_one.x / point_one.z,
        point_one.y / point_one.z,
        1.0,
    );
    let projected_two = Vector3::new(
        point_two.x / point_two.z,
        point_two.y / point_two.z,
        1.0,
    );
    let error_one = (projected_one.x - correspondence.x1.x)
        .hypot(projected_one.y - correspondence.x1.y);
    let error_two = (projected_two.x - correspondence.x2.x)
        .hypot(projected_two.y - correspondence.x2.y);
    (error_one + error_two) * 0.5 * focal_pixels
}

fn triangulation_angle(point: &Vector3<f64>, camera_two_center: &Vector3<f64>) -> f64 {
    let ray_one = point.normalize();
    let ray_two = (*point - camera_two_center).normalize();
    ray_one
        .dot(&ray_two)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn median_f64(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    values.sort_by(f64::total_cmp);
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

    fn project(
        point: Vector3<f64>,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) -> Vector3<f64> {
        let camera_point = rotation * point + translation;
        Vector3::new(
            camera_point.x / camera_point.z,
            camera_point.y / camera_point.z,
            1.0,
        )
    }

    fn feature_from_bearing(
        bearing: Vector3<f64>,
        width: u32,
        height: u32,
        focal: f64,
        descriptor: i16,
    ) -> Feature {
        let x = (bearing.x * focal + width as f64 * 0.5).round();
        let y = (bearing.y * focal + height as f64 * 0.5).round();
        Feature {
            x: x.clamp(0.0, width.saturating_sub(1) as f64) as u32,
            y: y.clamp(0.0, height.saturating_sub(1) as f64) as u32,
            score: 1.0,
            descriptor: vec![descriptor],
        }
    }

    fn synthetic_correspondences(
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) -> (Vec<Feature>, Vec<Feature>, Vec<FeatureMatch>) {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let mut source = Vec::new();
        let mut target = Vec::new();
        let mut matches = Vec::new();

        for index in 0..30 {
            let point = Vector3::new(
                (index % 6) as f64 * 0.36 - 0.9,
                (index / 6) as f64 * 0.28 - 0.56,
                3.2 + (index % 5) as f64 * 0.42,
            );
            let x1 = project(point, Matrix3::identity(), Vector3::zeros());
            let x2 = project(point, rotation, translation);
            source.push(feature_from_bearing(
                x1,
                width,
                height,
                focal,
                index as i16,
            ));
            target.push(feature_from_bearing(
                x2,
                width,
                height,
                focal,
                index as i16,
            ));
            matches.push(FeatureMatch {
                a: index,
                b: index,
                distance: 0.2,
            });
        }

        (source, target, matches)
    }

    #[test]
    fn triangulates_known_two_view_point() {
        let rotation = Matrix3::identity();
        let translation = Vector3::new(-1.0, 0.0, 0.0);
        let point = Vector3::new(0.4, -0.2, 4.0);
        let x1 = project(point, Matrix3::identity(), Vector3::zeros());
        let x2 = project(point, rotation, translation);
        let recovered =
            triangulate(&x1, &x2, &rotation, &translation).expect("triangulation");
        assert!((recovered - point).norm() < 1.0e-8);
    }

    #[test]
    fn recovers_translating_two_view_fixture_with_outliers() {
        let yaw = 0.07f64;
        let rotation = Matrix3::new(
            yaw.cos(),
            0.0,
            yaw.sin(),
            0.0,
            1.0,
            0.0,
            -yaw.sin(),
            0.0,
            yaw.cos(),
        );
        let translation = Vector3::new(-0.72, 0.03, 0.04);
        let (source, mut target, matches) = synthetic_correspondences(rotation, translation);
        for feature in target.iter_mut().skip(26) {
            feature.x = feature.x.saturating_add(70).min(635);
            feature.y = feature.y.saturating_sub(45);
        }

        let estimate = estimate_two_view(&source, &target, &matches, 640, 480, 500.0)
            .expect("calibrated two-view estimate");
        assert!(estimate.inliers >= 24, "inliers: {}", estimate.inliers);
        assert!(
            estimate.points.len() >= 20,
            "points: {}",
            estimate.points.len()
        );
        assert!(estimate.median_reprojection_error_pixels < 1.5);
        assert!(estimate.median_triangulation_angle_degrees > 0.5);

        let rotation_delta = estimate.rotation * rotation.transpose();
        let cosine = ((rotation_delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        assert!(cosine.acos() < 0.08);
        assert!(estimate.translation.dot(&translation.normalize()) > 0.85);
    }

    #[test]
    fn pure_rotation_is_rejected_as_two_view_baseline() {
        let yaw = 0.09f64;
        let rotation = Matrix3::new(
            yaw.cos(),
            0.0,
            yaw.sin(),
            0.0,
            1.0,
            0.0,
            -yaw.sin(),
            0.0,
            yaw.cos(),
        );
        let (source, target, matches) = synthetic_correspondences(rotation, Vector3::zeros());
        assert!(estimate_two_view(&source, &target, &matches, 640, 480, 500.0).is_none());
    }
}
