use super::{
    BundleAdjustmentResult, BundleAdjustmentStats, LandmarkSource, MultiViewAnalysis, NewLandmark,
    RegisteredCamera,
};
use crate::Feature;
use nalgebra::{Matrix3, SMatrix, SVector, Vector2, Vector3};
use std::collections::HashMap;

const MAX_ITERATIONS: usize = 8;
const MIN_CAMERA_OBSERVATIONS: usize = 8;
const MIN_LANDMARK_OBSERVATIONS: usize = 2;
const MAX_INPUT_REPROJECTION_ERROR_PIXELS: f64 = 4.0;
const HUBER_DELTA_PIXELS: f64 = 2.0;
const DAMPING_FACTOR: f64 = 1.0e-4;
const MAX_LANDMARK_STEP: f64 = 0.2;
const MAX_CAMERA_ROTATION_STEP_RADIANS: f64 = 0.04;
const MAX_CAMERA_TRANSLATION_STEP: f64 = 0.12;
const CONVERGENCE_EPSILON: f64 = 1.0e-6;
const MAX_MEDIAN_REGRESSION_PIXELS: f64 = 0.05;

type Matrix6 = SMatrix<f64, 6, 6>;
type Vector6 = SVector<f64, 6>;

#[derive(Clone, Copy, Debug)]
struct BundleObservation {
    camera_index: usize,
    landmark_index: usize,
    x_pixels: f64,
    y_pixels: f64,
}

#[derive(Clone, Debug)]
struct LandmarkState {
    position: Vector3<f64>,
    track_index: usize,
    source: LandmarkSource,
}

pub(super) fn optimize(
    analysis: &MultiViewAnalysis,
    seed_points: &[Vector3<f64>],
    new_landmarks: &[NewLandmark],
    cameras: &[RegisteredCamera],
    features: &[Vec<Feature>],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> BundleAdjustmentResult {
    let original_seed_points = seed_points.to_vec();
    let original_new_points: Vec<Vector3<f64>> = new_landmarks
        .iter()
        .map(|landmark| landmark.position)
        .collect();
    let original_cameras = cameras.to_vec();
    let mut stats = BundleAdjustmentStats::default();

    let Some(seed_pair_index) = analysis.seed_pair_index else {
        return BundleAdjustmentResult {
            stats,
            cameras: original_cameras,
            seed_points: original_seed_points,
            new_landmark_positions: original_new_points,
        };
    };
    if cameras.len() < 3 || !focal_pixels.is_finite() || focal_pixels <= 0.0 {
        return BundleAdjustmentResult {
            stats,
            cameras: original_cameras,
            seed_points: original_seed_points,
            new_landmark_positions: original_new_points,
        };
    }

    let mut landmarks = Vec::new();
    for seed_track in &analysis.seed_tracks {
        let Some(&position) = seed_points.get(seed_track.point_index) else {
            continue;
        };
        landmarks.push(LandmarkState {
            position,
            track_index: seed_track.track_index,
            source: LandmarkSource::Seed(seed_track.point_index),
        });
    }
    for (new_index, landmark) in new_landmarks.iter().enumerate() {
        landmarks.push(LandmarkState {
            position: landmark.position,
            track_index: landmark.track_index,
            source: LandmarkSource::New(new_index),
        });
    }

    let camera_by_frame: HashMap<usize, usize> = cameras
        .iter()
        .enumerate()
        .map(|(index, camera)| (camera.frame_index, index))
        .collect();
    let observations = build_observations(
        analysis,
        &landmarks,
        cameras,
        &camera_by_frame,
        features,
        width,
        height,
        focal_pixels,
    );
    let landmark_observations = observation_indices_by_landmark(landmarks.len(), &observations);
    let camera_observations = observation_indices_by_camera(cameras.len(), &observations);
    let fixed_frames = [seed_pair_index, seed_pair_index + 1];
    let optimizable_landmarks = landmark_observations
        .iter()
        .filter(|indices| indices.len() >= MIN_LANDMARK_OBSERVATIONS)
        .count();
    let optimizable_cameras = cameras
        .iter()
        .enumerate()
        .filter(|(index, camera)| {
            !fixed_frames.contains(&camera.frame_index)
                && camera_observations[*index].len() >= MIN_CAMERA_OBSERVATIONS
        })
        .count();

    stats.observations = observations.len();
    stats.optimized_landmarks = optimizable_landmarks;
    stats.optimized_cameras = optimizable_cameras;
    if optimizable_landmarks == 0 || optimizable_cameras == 0 || observations.is_empty() {
        return BundleAdjustmentResult {
            stats,
            cameras: original_cameras,
            seed_points: original_seed_points,
            new_landmark_positions: original_new_points,
        };
    }
    stats.attempted = true;

    let Some(initial_metrics) = error_metrics(
        cameras,
        &landmarks,
        &observations,
        width,
        height,
        focal_pixels,
    ) else {
        return BundleAdjustmentResult {
            stats,
            cameras: original_cameras,
            seed_points: original_seed_points,
            new_landmark_positions: original_new_points,
        };
    };
    stats.initial_median_reprojection_error_pixels = Some(initial_metrics.median as f32);
    stats.initial_rmse_reprojection_error_pixels = Some(initial_metrics.rmse as f32);

    let mut working_cameras = cameras.to_vec();
    let mut working_landmarks = landmarks.clone();
    let iterations = run_iterations(
        &mut working_cameras,
        &mut working_landmarks,
        &observations,
        &camera_observations,
        &landmark_observations,
        &fixed_frames,
        width,
        height,
        focal_pixels,
    );
    stats.iterations = iterations;

    let Some(final_metrics) = error_metrics(
        &working_cameras,
        &working_landmarks,
        &observations,
        width,
        height,
        focal_pixels,
    ) else {
        return BundleAdjustmentResult {
            stats,
            cameras: original_cameras,
            seed_points: original_seed_points,
            new_landmark_positions: original_new_points,
        };
    };
    stats.final_median_reprojection_error_pixels = Some(final_metrics.median as f32);
    stats.final_rmse_reprojection_error_pixels = Some(final_metrics.rmse as f32);

    let improved_cost = final_metrics.cost + 1.0e-8 < initial_metrics.cost;
    let improved_rmse = final_metrics.rmse <= initial_metrics.rmse + 1.0e-9;
    let median_stable =
        final_metrics.median <= initial_metrics.median + MAX_MEDIAN_REGRESSION_PIXELS;
    stats.accepted = improved_cost && improved_rmse && median_stable;

    if !stats.accepted {
        return BundleAdjustmentResult {
            stats,
            cameras: original_cameras,
            seed_points: original_seed_points,
            new_landmark_positions: original_new_points,
        };
    }

    let mut adjusted_seed_points = original_seed_points;
    let mut adjusted_new_points = original_new_points;
    for landmark in working_landmarks {
        match landmark.source {
            LandmarkSource::Seed(point_index) => {
                if let Some(position) = adjusted_seed_points.get_mut(point_index) {
                    *position = landmark.position;
                }
            }
            LandmarkSource::New(new_index) => {
                if let Some(position) = adjusted_new_points.get_mut(new_index) {
                    *position = landmark.position;
                }
            }
        }
    }

    BundleAdjustmentResult {
        stats,
        cameras: working_cameras,
        seed_points: adjusted_seed_points,
        new_landmark_positions: adjusted_new_points,
    }
}

fn build_observations(
    analysis: &MultiViewAnalysis,
    landmarks: &[LandmarkState],
    cameras: &[RegisteredCamera],
    camera_by_frame: &HashMap<usize, usize>,
    features: &[Vec<Feature>],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Vec<BundleObservation> {
    let mut observations = Vec::new();
    for (landmark_index, landmark) in landmarks.iter().enumerate() {
        let Some(track) = analysis.tracks.get(landmark.track_index) else {
            continue;
        };
        for track_observation in &track.observations {
            let Some(&camera_index) = camera_by_frame.get(&track_observation.frame_index) else {
                continue;
            };
            let Some(feature) = features
                .get(track_observation.frame_index)
                .and_then(|frame_features| frame_features.get(track_observation.feature_index))
            else {
                continue;
            };
            let observation = BundleObservation {
                camera_index,
                landmark_index,
                x_pixels: feature.x as f64,
                y_pixels: feature.y as f64,
            };
            let Some(error) = reprojection_error(
                &cameras[camera_index],
                &landmark.position,
                &observation,
                width,
                height,
                focal_pixels,
            ) else {
                continue;
            };
            if error <= MAX_INPUT_REPROJECTION_ERROR_PIXELS {
                observations.push(observation);
            }
        }
    }
    observations
}

fn observation_indices_by_landmark(
    landmark_count: usize,
    observations: &[BundleObservation],
) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); landmark_count];
    for (index, observation) in observations.iter().enumerate() {
        result[observation.landmark_index].push(index);
    }
    result
}

fn observation_indices_by_camera(
    camera_count: usize,
    observations: &[BundleObservation],
) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); camera_count];
    for (index, observation) in observations.iter().enumerate() {
        result[observation.camera_index].push(index);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn run_iterations(
    cameras: &mut [RegisteredCamera],
    landmarks: &mut [LandmarkState],
    observations: &[BundleObservation],
    camera_observations: &[Vec<usize>],
    landmark_observations: &[Vec<usize>],
    fixed_frames: &[usize; 2],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> usize {
    let Some(mut previous_cost) = total_cost(
        cameras,
        landmarks,
        observations,
        width,
        height,
        focal_pixels,
    ) else {
        return 0;
    };
    let mut completed_iterations = 0;

    for iteration in 0..MAX_ITERATIONS {
        let mut accepted_blocks = 0usize;
        for (landmark_index, observation_indices) in landmark_observations.iter().enumerate() {
            if observation_indices.len() < MIN_LANDMARK_OBSERVATIONS {
                continue;
            }
            if refine_landmark(
                landmark_index,
                landmarks,
                cameras,
                observations,
                observation_indices,
                width,
                height,
                focal_pixels,
            ) {
                accepted_blocks += 1;
            }
        }

        for (camera_index, observation_indices) in camera_observations.iter().enumerate() {
            if fixed_frames.contains(&cameras[camera_index].frame_index)
                || observation_indices.len() < MIN_CAMERA_OBSERVATIONS
            {
                continue;
            }
            if refine_camera(
                camera_index,
                cameras,
                landmarks,
                observations,
                observation_indices,
                width,
                height,
                focal_pixels,
            ) {
                accepted_blocks += 1;
            }
        }

        completed_iterations = iteration + 1;
        let Some(current_cost) = total_cost(
            cameras,
            landmarks,
            observations,
            width,
            height,
            focal_pixels,
        ) else {
            break;
        };
        let improvement = previous_cost - current_cost;
        if accepted_blocks == 0
            || improvement <= CONVERGENCE_EPSILON * (1.0 + previous_cost.abs())
        {
            break;
        }
        previous_cost = current_cost;
    }

    completed_iterations
}

#[allow(clippy::too_many_arguments)]
fn refine_landmark(
    landmark_index: usize,
    landmarks: &mut [LandmarkState],
    cameras: &[RegisteredCamera],
    observations: &[BundleObservation],
    observation_indices: &[usize],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> bool {
    let current_position = landmarks[landmark_index].position;
    let Some(current_cost) = landmark_cost(
        &current_position,
        cameras,
        observations,
        observation_indices,
        width,
        height,
        focal_pixels,
    ) else {
        return false;
    };

    let mut hessian = Matrix3::<f64>::zeros();
    let mut gradient = Vector3::<f64>::zeros();
    for &observation_index in observation_indices {
        let observation = &observations[observation_index];
        let camera = &cameras[observation.camera_index];
        let Some((residual, camera_point)) = residual_and_camera_point(
            camera,
            &current_position,
            observation,
            width,
            height,
            focal_pixels,
        ) else {
            return false;
        };
        let weight = huber_weight(residual.norm());
        let (gradient_u_camera, gradient_v_camera) =
            projection_gradients(&camera_point, focal_pixels);
        let gradient_u = camera.rotation.transpose() * gradient_u_camera;
        let gradient_v = camera.rotation.transpose() * gradient_v_camera;
        hessian += weight
            * (gradient_u * gradient_u.transpose() + gradient_v * gradient_v.transpose());
        gradient += weight * (gradient_u * residual.x + gradient_v * residual.y);
    }

    add_damping3(&mut hessian);
    let Some(mut delta) = hessian.lu().solve(&(-gradient)) else {
        return false;
    };
    clamp_vector3(&mut delta, MAX_LANDMARK_STEP);
    if !delta.iter().all(|value| value.is_finite()) || delta.norm() <= 1.0e-10 {
        return false;
    }

    let candidate = current_position + delta;
    let Some(candidate_cost) = landmark_cost(
        &candidate,
        cameras,
        observations,
        observation_indices,
        width,
        height,
        focal_pixels,
    ) else {
        return false;
    };
    if candidate_cost + 1.0e-10 < current_cost {
        landmarks[landmark_index].position = candidate;
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_camera(
    camera_index: usize,
    cameras: &mut [RegisteredCamera],
    landmarks: &[LandmarkState],
    observations: &[BundleObservation],
    observation_indices: &[usize],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> bool {
    let current_camera = cameras[camera_index].clone();
    let Some(current_cost) = camera_cost(
        &current_camera,
        landmarks,
        observations,
        observation_indices,
        width,
        height,
        focal_pixels,
    ) else {
        return false;
    };

    let mut hessian = Matrix6::zeros();
    let mut gradient = Vector6::zeros();
    for &observation_index in observation_indices {
        let observation = &observations[observation_index];
        let landmark = &landmarks[observation.landmark_index];
        let Some((residual, camera_point)) = residual_and_camera_point(
            &current_camera,
            &landmark.position,
            observation,
            width,
            height,
            focal_pixels,
        ) else {
            return false;
        };
        let weight = huber_weight(residual.norm());
        let (gradient_u_camera, gradient_v_camera) =
            projection_gradients(&camera_point, focal_pixels);
        let gradient_u_rotation = skew(&camera_point) * gradient_u_camera;
        let gradient_v_rotation = skew(&camera_point) * gradient_v_camera;
        let jacobian_u = Vector6::new(
            gradient_u_rotation.x,
            gradient_u_rotation.y,
            gradient_u_rotation.z,
            gradient_u_camera.x,
            gradient_u_camera.y,
            gradient_u_camera.z,
        );
        let jacobian_v = Vector6::new(
            gradient_v_rotation.x,
            gradient_v_rotation.y,
            gradient_v_rotation.z,
            gradient_v_camera.x,
            gradient_v_camera.y,
            gradient_v_camera.z,
        );
        hessian += weight
            * (jacobian_u * jacobian_u.transpose()
                + jacobian_v * jacobian_v.transpose());
        gradient += weight * (jacobian_u * residual.x + jacobian_v * residual.y);
    }

    add_damping6(&mut hessian);
    let Some(mut delta) = hessian.lu().solve(&(-gradient)) else {
        return false;
    };
    let mut rotation_delta = Vector3::new(delta[0], delta[1], delta[2]);
    let mut translation_delta = Vector3::new(delta[3], delta[4], delta[5]);
    clamp_vector3(&mut rotation_delta, MAX_CAMERA_ROTATION_STEP_RADIANS);
    clamp_vector3(&mut translation_delta, MAX_CAMERA_TRANSLATION_STEP);
    for index in 0..3 {
        delta[index] = rotation_delta[index];
        delta[index + 3] = translation_delta[index];
    }
    if !delta.iter().all(|value| value.is_finite()) || delta.norm() <= 1.0e-10 {
        return false;
    }

    let rotation_increment = rotation_from_vector(&rotation_delta);
    let candidate_camera = RegisteredCamera {
        frame_index: current_camera.frame_index,
        rotation: rotation_increment * current_camera.rotation,
        translation: rotation_increment * current_camera.translation + translation_delta,
    };
    let Some(candidate_cost) = camera_cost(
        &candidate_camera,
        landmarks,
        observations,
        observation_indices,
        width,
        height,
        focal_pixels,
    ) else {
        return false;
    };
    if candidate_cost + 1.0e-10 < current_cost {
        cameras[camera_index] = candidate_camera;
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn landmark_cost(
    position: &Vector3<f64>,
    cameras: &[RegisteredCamera],
    observations: &[BundleObservation],
    observation_indices: &[usize],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<f64> {
    let mut cost = 0.0;
    for &observation_index in observation_indices {
        let observation = &observations[observation_index];
        let error = reprojection_error(
            &cameras[observation.camera_index],
            position,
            observation,
            width,
            height,
            focal_pixels,
        )?;
        cost += huber_cost(error);
    }
    Some(cost)
}

#[allow(clippy::too_many_arguments)]
fn camera_cost(
    camera: &RegisteredCamera,
    landmarks: &[LandmarkState],
    observations: &[BundleObservation],
    observation_indices: &[usize],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<f64> {
    let mut cost = 0.0;
    for &observation_index in observation_indices {
        let observation = &observations[observation_index];
        let error = reprojection_error(
            camera,
            &landmarks[observation.landmark_index].position,
            observation,
            width,
            height,
            focal_pixels,
        )?;
        cost += huber_cost(error);
    }
    Some(cost)
}

#[derive(Clone, Copy)]
struct ErrorMetrics {
    cost: f64,
    median: f64,
    rmse: f64,
}

fn error_metrics(
    cameras: &[RegisteredCamera],
    landmarks: &[LandmarkState],
    observations: &[BundleObservation],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<ErrorMetrics> {
    if observations.is_empty() {
        return None;
    }
    let mut errors = Vec::with_capacity(observations.len());
    let mut squared_sum = 0.0;
    let mut cost = 0.0;
    for observation in observations {
        let error = reprojection_error(
            &cameras[observation.camera_index],
            &landmarks[observation.landmark_index].position,
            observation,
            width,
            height,
            focal_pixels,
        )?;
        squared_sum += error * error;
        cost += huber_cost(error);
        errors.push(error);
    }
    let median = median_f64(&mut errors)?;
    Some(ErrorMetrics {
        cost,
        median,
        rmse: (squared_sum / observations.len() as f64).sqrt(),
    })
}

fn total_cost(
    cameras: &[RegisteredCamera],
    landmarks: &[LandmarkState],
    observations: &[BundleObservation],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<f64> {
    error_metrics(
        cameras,
        landmarks,
        observations,
        width,
        height,
        focal_pixels,
    )
    .map(|metrics| metrics.cost)
}

fn reprojection_error(
    camera: &RegisteredCamera,
    position: &Vector3<f64>,
    observation: &BundleObservation,
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<f64> {
    residual_and_camera_point(camera, position, observation, width, height, focal_pixels)
        .map(|(residual, _)| residual.norm())
}

fn residual_and_camera_point(
    camera: &RegisteredCamera,
    position: &Vector3<f64>,
    observation: &BundleObservation,
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<(Vector2<f64>, Vector3<f64>)> {
    let camera_point = camera.rotation * position + camera.translation;
    if camera_point.z <= 1.0e-6 || !camera_point.iter().all(|value| value.is_finite()) {
        return None;
    }
    let projected_x = camera_point.x / camera_point.z * focal_pixels + width as f64 * 0.5;
    let projected_y = camera_point.y / camera_point.z * focal_pixels + height as f64 * 0.5;
    let residual = Vector2::new(
        projected_x - observation.x_pixels,
        projected_y - observation.y_pixels,
    );
    residual
        .iter()
        .all(|value| value.is_finite())
        .then_some((residual, camera_point))
}

fn projection_gradients(
    camera_point: &Vector3<f64>,
    focal_pixels: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let inverse_z = 1.0 / camera_point.z;
    let inverse_z_squared = inverse_z * inverse_z;
    (
        Vector3::new(
            focal_pixels * inverse_z,
            0.0,
            -focal_pixels * camera_point.x * inverse_z_squared,
        ),
        Vector3::new(
            0.0,
            focal_pixels * inverse_z,
            -focal_pixels * camera_point.y * inverse_z_squared,
        ),
    )
}

fn skew(vector: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -vector.z, vector.y, vector.z, 0.0, -vector.x, -vector.y, vector.x, 0.0,
    )
}

fn rotation_from_vector(vector: &Vector3<f64>) -> Matrix3<f64> {
    let theta = vector.norm();
    if theta <= 1.0e-12 {
        return Matrix3::identity();
    }
    let cross = skew(vector);
    let a = theta.sin() / theta;
    let b = (1.0 - theta.cos()) / (theta * theta);
    Matrix3::identity() + cross * a + cross * cross * b
}

fn add_damping3(matrix: &mut Matrix3<f64>) {
    let scale = (matrix.trace() / 3.0).abs().max(1.0) * DAMPING_FACTOR;
    for index in 0..3 {
        matrix[(index, index)] += scale;
    }
}

fn add_damping6(matrix: &mut Matrix6) {
    let scale = (matrix.trace() / 6.0).abs().max(1.0) * DAMPING_FACTOR;
    for index in 0..6 {
        matrix[(index, index)] += scale;
    }
}

fn clamp_vector3(vector: &mut Vector3<f64>, maximum_norm: f64) {
    let norm = vector.norm();
    if norm > maximum_norm && norm.is_finite() {
        *vector *= maximum_norm / norm;
    }
}

fn huber_weight(error: f64) -> f64 {
    if error <= HUBER_DELTA_PIXELS || error <= f64::EPSILON {
        1.0
    } else {
        HUBER_DELTA_PIXELS / error
    }
}

fn huber_cost(error: f64) -> f64 {
    if error <= HUBER_DELTA_PIXELS {
        0.5 * error * error
    } else {
        HUBER_DELTA_PIXELS * (error - 0.5 * HUBER_DELTA_PIXELS)
    }
}

fn median_f64(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(frame_index: usize, center: Vector3<f64>) -> RegisteredCamera {
        RegisteredCamera {
            frame_index,
            rotation: Matrix3::identity(),
            translation: -center,
        }
    }

    fn observe(
        camera: &RegisteredCamera,
        point: &Vector3<f64>,
        focal: f64,
        width: u32,
        height: u32,
    ) -> (f64, f64) {
        let camera_point = camera.rotation * point + camera.translation;
        (
            camera_point.x / camera_point.z * focal + width as f64 * 0.5,
            camera_point.y / camera_point.z * focal + height as f64 * 0.5,
        )
    }

    #[test]
    fn bundle_adjustment_reduces_error_without_moving_seed_gauge() {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let true_cameras = [
            camera(0, Vector3::new(0.0, 0.0, 0.0)),
            camera(1, Vector3::new(1.0, 0.0, 0.0)),
            camera(2, Vector3::new(2.0, 0.0, 0.0)),
        ];
        let mut cameras = vec![
            true_cameras[0].clone(),
            true_cameras[1].clone(),
            camera(2, Vector3::new(1.82, 0.07, -0.03)),
        ];
        let initial_fixed = [cameras[0].clone(), cameras[1].clone()];

        let true_points: Vec<Vector3<f64>> = (0..12)
            .map(|index| {
                Vector3::new(
                    -0.45 + (index % 4) as f64 * 0.3,
                    -0.3 + (index / 4) as f64 * 0.3,
                    4.0 + (index % 3) as f64 * 0.55,
                )
            })
            .collect();
        let mut landmarks: Vec<LandmarkState> = true_points
            .iter()
            .enumerate()
            .map(|(index, point)| LandmarkState {
                position: *point
                    + Vector3::new(
                        (index as f64 % 3.0 - 1.0) * 0.025,
                        (index as f64 % 2.0 - 0.5) * 0.02,
                        (index as f64 % 4.0 - 1.5) * 0.015,
                    ),
                track_index: index,
                source: LandmarkSource::Seed(index),
            })
            .collect();
        let mut observations = Vec::new();
        for (landmark_index, point) in true_points.iter().enumerate() {
            for (camera_index, true_camera) in true_cameras.iter().enumerate() {
                let (x_pixels, y_pixels) = observe(true_camera, point, focal, width, height);
                observations.push(BundleObservation {
                    camera_index,
                    landmark_index,
                    x_pixels,
                    y_pixels,
                });
            }
        }
        let camera_observations = observation_indices_by_camera(cameras.len(), &observations);
        let landmark_observations = observation_indices_by_landmark(landmarks.len(), &observations);
        let initial =
            error_metrics(&cameras, &landmarks, &observations, width, height, focal)
                .expect("initial geometry should project");
        let initial_camera_error =
            (cameras[2].camera_center() - true_cameras[2].camera_center()).norm();

        let iterations = run_iterations(
            &mut cameras,
            &mut landmarks,
            &observations,
            &camera_observations,
            &landmark_observations,
            &[0, 1],
            width,
            height,
            focal,
        );
        let final_metrics =
            error_metrics(&cameras, &landmarks, &observations, width, height, focal)
                .expect("adjusted geometry should project");
        let final_camera_error =
            (cameras[2].camera_center() - true_cameras[2].camera_center()).norm();

        assert!(iterations > 0);
        assert!(final_metrics.rmse < initial.rmse);
        assert!(final_metrics.cost < initial.cost);
        assert!(final_camera_error < initial_camera_error);
        assert_eq!(cameras[0].rotation, initial_fixed[0].rotation);
        assert_eq!(cameras[0].translation, initial_fixed[0].translation);
        assert_eq!(cameras[1].rotation, initial_fixed[1].rotation);
        assert_eq!(cameras[1].translation, initial_fixed[1].translation);
    }

    #[test]
    fn bundle_adjustment_keeps_perfect_geometry_unchanged() {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let mut cameras = vec![
            camera(0, Vector3::new(0.0, 0.0, 0.0)),
            camera(1, Vector3::new(1.0, 0.0, 0.0)),
            camera(2, Vector3::new(2.0, 0.0, 0.0)),
        ];
        let mut landmarks: Vec<LandmarkState> = (0..8)
            .map(|index| LandmarkState {
                position: Vector3::new(index as f64 * 0.08 - 0.28, 0.1, 4.0 + index as f64 * 0.1),
                track_index: index,
                source: LandmarkSource::Seed(index),
            })
            .collect();
        let original_cameras = cameras.clone();
        let original_positions: Vec<Vector3<f64>> =
            landmarks.iter().map(|landmark| landmark.position).collect();
        let mut observations = Vec::new();
        for (landmark_index, landmark) in landmarks.iter().enumerate() {
            for (camera_index, camera) in cameras.iter().enumerate() {
                let (x_pixels, y_pixels) =
                    observe(camera, &landmark.position, focal, width, height);
                observations.push(BundleObservation {
                    camera_index,
                    landmark_index,
                    x_pixels,
                    y_pixels,
                });
            }
        }
        let camera_observations = observation_indices_by_camera(cameras.len(), &observations);
        let landmark_observations = observation_indices_by_landmark(landmarks.len(), &observations);

        run_iterations(
            &mut cameras,
            &mut landmarks,
            &observations,
            &camera_observations,
            &landmark_observations,
            &[0, 1],
            width,
            height,
            focal,
        );

        for (camera, original) in cameras.iter().zip(&original_cameras) {
            assert!((camera.translation - original.translation).norm() < 1.0e-10);
            assert!((camera.rotation - original.rotation).norm() < 1.0e-10);
        }
        for (landmark, original) in landmarks.iter().zip(&original_positions) {
            assert!((landmark.position - original).norm() < 1.0e-10);
        }
    }
}
