use nalgebra::{Matrix3, Matrix4, Vector3};

const MIN_TRIANGULATION_ANGLE_DEGREES: f64 = 0.5;
const MAX_REPROJECTION_ERROR_PIXELS: f64 = 4.0;
const MIN_INLIER_RATIO_NUMERATOR: usize = 3;
const MIN_INLIER_RATIO_DENOMINATOR: usize = 5;

#[derive(Clone, Debug)]
pub(super) struct KnownCamera {
    pub frame_index: usize,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ImageObservation {
    pub frame_index: usize,
    pub x_pixels: f64,
    pub y_pixels: f64,
}

#[derive(Clone, Debug)]
pub(super) struct TrackTriangulation {
    pub position: Vector3<f64>,
    pub inliers: usize,
    pub observations: usize,
    pub median_reprojection_error_pixels: f64,
    pub triangulation_angle_degrees: f64,
}

pub(super) fn triangulate_track(
    observations: &[ImageObservation],
    cameras: &[KnownCamera],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<TrackTriangulation> {
    if observations.len() < 2 || cameras.len() < 2 || !focal_pixels.is_finite() || focal_pixels <= 0.0 {
        return None;
    }

    let supported: Vec<(ImageObservation, &KnownCamera)> = observations
        .iter()
        .filter_map(|&observation| {
            cameras
                .iter()
                .find(|camera| camera.frame_index == observation.frame_index)
                .map(|camera| (observation, camera))
        })
        .collect();
    if supported.len() < 2 {
        return None;
    }

    let center_x = width as f64 * 0.5;
    let center_y = height as f64 * 0.5;
    let mut best: Option<TrackTriangulation> = None;

    for left_index in 0..supported.len() - 1 {
        for right_index in left_index + 1..supported.len() {
            let (left_observation, left_camera) = supported[left_index];
            let (right_observation, right_camera) = supported[right_index];
            if left_camera.frame_index == right_camera.frame_index {
                continue;
            }

            let left_normalized = (
                (left_observation.x_pixels - center_x) / focal_pixels,
                (left_observation.y_pixels - center_y) / focal_pixels,
            );
            let right_normalized = (
                (right_observation.x_pixels - center_x) / focal_pixels,
                (right_observation.y_pixels - center_y) / focal_pixels,
            );
            let Some(position) = triangulate_pair(
                left_normalized,
                right_normalized,
                left_camera,
                right_camera,
            ) else {
                continue;
            };

            let angle = triangulation_angle(&position, left_camera, right_camera);
            if !angle.is_finite() || angle < MIN_TRIANGULATION_ANGLE_DEGREES {
                continue;
            }

            let mut inlier_errors = Vec::new();
            for (observation, camera) in &supported {
                let error = reprojection_error(
                    &position,
                    observation,
                    camera,
                    center_x,
                    center_y,
                    focal_pixels,
                );
                if error.is_finite() && error <= MAX_REPROJECTION_ERROR_PIXELS {
                    inlier_errors.push(error);
                }
            }

            let required_inliers = ((supported.len() * MIN_INLIER_RATIO_NUMERATOR)
                .div_ceil(MIN_INLIER_RATIO_DENOMINATOR))
            .max(2);
            if inlier_errors.len() < required_inliers {
                continue;
            }
            let median_error = median(&mut inlier_errors);
            let candidate = TrackTriangulation {
                position,
                inliers: inlier_errors.len(),
                observations: supported.len(),
                median_reprojection_error_pixels: median_error,
                triangulation_angle_degrees: angle,
            };
            if best.as_ref().is_none_or(|current| is_better(&candidate, current)) {
                best = Some(candidate);
            }
        }
    }

    best
}

fn is_better(candidate: &TrackTriangulation, current: &TrackTriangulation) -> bool {
    candidate.inliers > current.inliers
        || (candidate.inliers == current.inliers
            && (candidate.median_reprojection_error_pixels
                < current.median_reprojection_error_pixels
                || (candidate.median_reprojection_error_pixels
                    == current.median_reprojection_error_pixels
                    && candidate.triangulation_angle_degrees > current.triangulation_angle_degrees)))
}

fn triangulate_pair(
    left: (f64, f64),
    right: (f64, f64),
    left_camera: &KnownCamera,
    right_camera: &KnownCamera,
) -> Option<Vector3<f64>> {
    let mut system = Matrix4::<f64>::zeros();
    fill_projection_equations(&mut system, 0, left, left_camera);
    fill_projection_equations(&mut system, 2, right, right_camera);

    let svd = system.svd(false, true);
    let v_t = svd.v_t?;
    let homogeneous = v_t.row(3);
    let w = homogeneous[3];
    if !w.is_finite() || w.abs() <= 1.0e-12 {
        return None;
    }
    let point = Vector3::new(homogeneous[0] / w, homogeneous[1] / w, homogeneous[2] / w);
    point.iter().all(|value| value.is_finite()).then_some(point)
}

fn fill_projection_equations(
    system: &mut Matrix4<f64>,
    row: usize,
    normalized: (f64, f64),
    camera: &KnownCamera,
) {
    for column in 0..4 {
        let p0 = projection_element(camera, 0, column);
        let p1 = projection_element(camera, 1, column);
        let p2 = projection_element(camera, 2, column);
        system[(row, column)] = normalized.0 * p2 - p0;
        system[(row + 1, column)] = normalized.1 * p2 - p1;
    }
}

fn projection_element(camera: &KnownCamera, row: usize, column: usize) -> f64 {
    if column < 3 {
        camera.rotation[(row, column)]
    } else {
        camera.translation[row]
    }
}

fn camera_center(camera: &KnownCamera) -> Vector3<f64> {
    -camera.rotation.transpose() * camera.translation
}

fn triangulation_angle(
    point: &Vector3<f64>,
    left_camera: &KnownCamera,
    right_camera: &KnownCamera,
) -> f64 {
    let left_ray = *point - camera_center(left_camera);
    let right_ray = *point - camera_center(right_camera);
    if left_ray.norm_squared() <= 1.0e-12 || right_ray.norm_squared() <= 1.0e-12 {
        return 0.0;
    }
    left_ray
        .normalize()
        .dot(&right_ray.normalize())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn reprojection_error(
    point: &Vector3<f64>,
    observation: &ImageObservation,
    camera: &KnownCamera,
    center_x: f64,
    center_y: f64,
    focal_pixels: f64,
) -> f64 {
    let camera_point = camera.rotation * *point + camera.translation;
    if !camera_point.iter().all(|value| value.is_finite()) || camera_point.z <= 1.0e-6 {
        return f64::INFINITY;
    }
    let projected_x = focal_pixels * camera_point.x / camera_point.z + center_x;
    let projected_y = focal_pixels * camera_point.y / camera_point.z + center_y;
    (projected_x - observation.x_pixels).hypot(projected_y - observation.y_pixels)
}

fn median(values: &mut [f64]) -> f64 {
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

    fn camera(frame_index: usize, center_x: f64) -> KnownCamera {
        KnownCamera {
            frame_index,
            rotation: Matrix3::identity(),
            translation: Vector3::new(-center_x, 0.0, 0.0),
        }
    }

    fn observe(point: Vector3<f64>, camera: &KnownCamera, width: u32, height: u32, focal: f64) -> ImageObservation {
        let camera_point = camera.rotation * point + camera.translation;
        ImageObservation {
            frame_index: camera.frame_index,
            x_pixels: focal * camera_point.x / camera_point.z + width as f64 * 0.5,
            y_pixels: focal * camera_point.y / camera_point.z + height as f64 * 0.5,
        }
    }

    #[test]
    fn triangulates_a_track_from_multiple_accepted_views() {
        let width = 640;
        let height = 480;
        let focal = 520.0;
        let point = Vector3::new(0.35, -0.18, 4.2);
        let cameras = vec![camera(0, 0.0), camera(1, 0.8), camera(2, 1.4)];
        let observations: Vec<ImageObservation> = cameras
            .iter()
            .map(|camera| observe(point, camera, width, height, focal))
            .collect();

        let estimate = triangulate_track(&observations, &cameras, width, height, focal)
            .expect("track should triangulate");

        assert!((estimate.position - point).norm() < 1.0e-8);
        assert_eq!(estimate.inliers, 3);
        assert_eq!(estimate.observations, 3);
        assert!(estimate.median_reprojection_error_pixels < 1.0e-8);
        assert!(estimate.triangulation_angle_degrees > 5.0);
    }

    #[test]
    fn rejects_a_low_angle_track() {
        let width = 640;
        let height = 480;
        let focal = 520.0;
        let point = Vector3::new(0.0, 0.0, 20.0);
        let cameras = vec![camera(0, 0.0), camera(1, 0.01)];
        let observations: Vec<ImageObservation> = cameras
            .iter()
            .map(|camera| observe(point, camera, width, height, focal))
            .collect();

        assert!(triangulate_track(&observations, &cameras, width, height, focal).is_none());
    }

    #[test]
    fn tolerates_one_outlier_when_two_views_support_the_track() {
        let width = 640;
        let height = 480;
        let focal = 520.0;
        let point = Vector3::new(-0.25, 0.12, 3.8);
        let cameras = vec![camera(0, 0.0), camera(1, 0.9), camera(2, 1.5)];
        let mut observations: Vec<ImageObservation> = cameras
            .iter()
            .map(|camera| observe(point, camera, width, height, focal))
            .collect();
        observations[2].x_pixels += 70.0;
        observations[2].y_pixels -= 40.0;

        let estimate = triangulate_track(&observations, &cameras, width, height, focal)
            .expect("two agreeing views should survive one outlier");

        assert_eq!(estimate.inliers, 2);
        assert!((estimate.position - point).norm() < 1.0e-8);
    }
}
