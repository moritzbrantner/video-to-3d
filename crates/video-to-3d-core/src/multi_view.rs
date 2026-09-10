use super::{Feature, FeatureMatch, PairStats};
use nalgebra::{Matrix3, Matrix4, Vector3};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

const MIN_KEYFRAME_OVERLAP: f32 = 0.18;
const KEYFRAME_PARALLAX_BUDGET: f32 = 1.25;
const MIN_STRONG_PAIR_PARALLAX: f32 = 0.55;
const MIN_PNP_CORRESPONDENCES: usize = 8;
const MIN_NEW_LANDMARK_TRIANGULATION_ANGLE_DEGREES: f64 = 0.5;
const MAX_NEW_LANDMARK_REPROJECTION_ERROR_PIXELS: f64 = 4.0;
const MAX_NEW_LANDMARK_MEDIAN_REPROJECTION_ERROR_PIXELS: f64 = 2.5;
const MIN_NEW_LANDMARK_SUPPORT_RATIO: f64 = 0.6;

#[derive(Clone, Debug, Serialize)]
pub struct RegistrationCandidateStats {
    pub frame_index: usize,
    pub seed_landmark_correspondences: usize,
    pub pnp_ready: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct NewLandmarkStats {
    pub candidate_tracks: usize,
    pub accepted_landmarks: usize,
    pub supporting_observations: usize,
    pub median_reprojection_error_pixels: Option<f32>,
    pub median_triangulation_angle_degrees: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MultiViewStats {
    pub keyframes: Vec<usize>,
    pub track_count: usize,
    pub tracks_three_plus: usize,
    pub longest_track: usize,
    pub observations: usize,
    pub linked_pairs: usize,
    pub registration_candidates: Vec<RegistrationCandidateStats>,
    pub new_landmarks: NewLandmarkStats,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SeedLandmark {
    pub point_index: usize,
    pub source_feature_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RegistrationCorrespondence {
    pub seed_point_index: usize,
    pub feature_index: usize,
}

#[derive(Clone, Debug)]
pub(super) struct RegistrationCandidate {
    pub frame_index: usize,
    pub correspondences: Vec<RegistrationCorrespondence>,
}

#[derive(Clone, Debug)]
pub(super) struct RegisteredCamera {
    pub frame_index: usize,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

impl RegisteredCamera {
    fn camera_center(&self) -> Vector3<f64> {
        -self.rotation.transpose() * self.translation
    }
}

#[derive(Clone, Debug)]
pub(super) struct NewLandmark {
    pub position: Vector3<f64>,
    pub source_frame_index: usize,
    pub source_feature_index: usize,
    pub supporting_observations: usize,
    pub median_reprojection_error_pixels: f64,
    pub triangulation_angle_degrees: f64,
}

#[derive(Clone, Debug)]
pub(super) struct NewLandmarkAnalysis {
    pub stats: NewLandmarkStats,
    pub landmarks: Vec<NewLandmark>,
}

#[derive(Clone, Debug)]
pub(super) struct MultiViewAnalysis {
    pub stats: MultiViewStats,
    pub registration_candidates: Vec<RegistrationCandidate>,
    tracks: Vec<FeatureTrack>,
    seed_track_indices: HashSet<usize>,
    seed_pair_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Observation {
    frame_index: usize,
    feature_index: usize,
}

#[derive(Clone, Debug)]
struct FeatureTrack {
    observations: Vec<Observation>,
}

#[derive(Clone, Copy)]
struct RegisteredObservation<'a> {
    camera: &'a RegisteredCamera,
    frame_index: usize,
    feature_index: usize,
    x_pixels: f64,
    y_pixels: f64,
}

#[derive(Clone, Debug)]
struct TriangulatedTrack {
    position: Vector3<f64>,
    source_frame_index: usize,
    source_feature_index: usize,
    supporting_observations: usize,
    median_reprojection_error_pixels: f64,
    triangulation_angle_degrees: f64,
}

pub(super) fn analyze(
    pairs: &[PairStats],
    adjacent_matches: &[Vec<FeatureMatch>],
    seed_pair: Option<(usize, &[SeedLandmark])>,
) -> MultiViewAnalysis {
    let (tracks, membership) = build_feature_tracks(adjacent_matches);
    let keyframes = select_keyframes(pairs);
    let seed_pair_index = seed_pair.map(|(pair_index, _)| pair_index);
    let seed_track_indices = seed_pair
        .map(|(pair_index, seed_landmarks)| {
            collect_seed_track_indices(&membership, pair_index, seed_landmarks)
        })
        .unwrap_or_default();
    let registration_candidates = seed_pair
        .map(|(pair_index, seed_landmarks)| {
            registration_candidates(&keyframes, &tracks, &membership, pair_index, seed_landmarks)
        })
        .unwrap_or_default();
    let registration_candidate_stats = registration_candidates
        .iter()
        .map(|candidate| RegistrationCandidateStats {
            frame_index: candidate.frame_index,
            seed_landmark_correspondences: candidate.correspondences.len(),
            pnp_ready: candidate.correspondences.len() >= MIN_PNP_CORRESPONDENCES,
        })
        .collect();

    MultiViewAnalysis {
        stats: MultiViewStats {
            keyframes,
            track_count: tracks.len(),
            tracks_three_plus: tracks
                .iter()
                .filter(|track| track.observations.len() >= 3)
                .count(),
            longest_track: tracks
                .iter()
                .map(|track| track.observations.len())
                .max()
                .unwrap_or(0),
            observations: tracks.iter().map(|track| track.observations.len()).sum(),
            linked_pairs: adjacent_matches
                .iter()
                .filter(|matches| !matches.is_empty())
                .count(),
            registration_candidates: registration_candidate_stats,
            new_landmarks: NewLandmarkStats::default(),
        },
        registration_candidates,
        tracks,
        seed_track_indices,
        seed_pair_index,
    }
}

pub(super) fn triangulate_new_landmarks(
    analysis: &MultiViewAnalysis,
    cameras: &[RegisteredCamera],
    features: &[Vec<Feature>],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> NewLandmarkAnalysis {
    let Some(seed_pair_index) = analysis.seed_pair_index else {
        return NewLandmarkAnalysis {
            stats: NewLandmarkStats::default(),
            landmarks: Vec::new(),
        };
    };
    if cameras.len() < 3 || !focal_pixels.is_finite() || focal_pixels <= 0.0 {
        return NewLandmarkAnalysis {
            stats: NewLandmarkStats::default(),
            landmarks: Vec::new(),
        };
    }

    let seed_frames = [seed_pair_index, seed_pair_index + 1];
    let camera_by_frame: HashMap<usize, &RegisteredCamera> = cameras
        .iter()
        .map(|camera| (camera.frame_index, camera))
        .collect();
    let mut candidate_tracks = 0usize;
    let mut landmarks = Vec::new();
    let mut reprojection_errors = Vec::new();
    let mut triangulation_angles = Vec::new();

    for (track_index, track) in analysis.tracks.iter().enumerate() {
        if analysis.seed_track_indices.contains(&track_index) {
            continue;
        }

        let observations: Vec<RegisteredObservation<'_>> = track
            .observations
            .iter()
            .filter_map(|observation| {
                let camera = camera_by_frame.get(&observation.frame_index).copied()?;
                let feature = features
                    .get(observation.frame_index)?
                    .get(observation.feature_index)?;
                Some(RegisteredObservation {
                    camera,
                    frame_index: observation.frame_index,
                    feature_index: observation.feature_index,
                    x_pixels: feature.x as f64,
                    y_pixels: feature.y as f64,
                })
            })
            .collect();

        if observations.len() < 2
            || !observations
                .iter()
                .any(|observation| !seed_frames.contains(&observation.frame_index))
        {
            continue;
        }
        candidate_tracks += 1;

        let Some(triangulated) =
            triangulate_track(&observations, &seed_frames, width, height, focal_pixels)
        else {
            continue;
        };

        reprojection_errors.push(triangulated.median_reprojection_error_pixels);
        triangulation_angles.push(triangulated.triangulation_angle_degrees);
        landmarks.push(NewLandmark {
            position: triangulated.position,
            source_frame_index: triangulated.source_frame_index,
            source_feature_index: triangulated.source_feature_index,
            supporting_observations: triangulated.supporting_observations,
            median_reprojection_error_pixels: triangulated.median_reprojection_error_pixels,
            triangulation_angle_degrees: triangulated.triangulation_angle_degrees,
        });
    }

    let supporting_observations = landmarks
        .iter()
        .map(|landmark| landmark.supporting_observations)
        .sum();
    let stats = NewLandmarkStats {
        candidate_tracks,
        accepted_landmarks: landmarks.len(),
        supporting_observations,
        median_reprojection_error_pixels: median_f64(&mut reprojection_errors)
            .map(|value| value as f32),
        median_triangulation_angle_degrees: median_f64(&mut triangulation_angles)
            .map(|value| value as f32),
    };

    NewLandmarkAnalysis { stats, landmarks }
}

fn collect_seed_track_indices(
    membership: &HashMap<Observation, usize>,
    seed_pair_index: usize,
    seed_landmarks: &[SeedLandmark],
) -> HashSet<usize> {
    seed_landmarks
        .iter()
        .filter_map(|seed_landmark| {
            membership
                .get(&Observation {
                    frame_index: seed_pair_index,
                    feature_index: seed_landmark.source_feature_index,
                })
                .copied()
        })
        .collect()
}

fn build_feature_tracks(
    adjacent_matches: &[Vec<FeatureMatch>],
) -> (Vec<FeatureTrack>, HashMap<Observation, usize>) {
    let mut tracks: Vec<FeatureTrack> = Vec::new();
    let mut membership: HashMap<Observation, usize> = HashMap::new();

    for (pair_index, matches) in adjacent_matches.iter().enumerate() {
        for feature_match in matches {
            let source = Observation {
                frame_index: pair_index,
                feature_index: feature_match.a,
            };
            let target = Observation {
                frame_index: pair_index + 1,
                feature_index: feature_match.b,
            };

            if let Some(&track_index) = membership.get(&source) {
                match membership.get(&target).copied() {
                    Some(existing_track) if existing_track != track_index => {
                        // Adjacent matching is one-to-one, so this should not occur. Keep the
                        // earlier deterministic assignment rather than merging ambiguous tracks.
                    }
                    Some(_) => {}
                    None => {
                        tracks[track_index].observations.push(target);
                        membership.insert(target, track_index);
                    }
                }
            } else if membership.contains_key(&target) {
                // A target can only be claimed once by the one-to-one matcher. Ignore a
                // conflicting observation rather than creating a non-deterministic merge.
                continue;
            } else {
                let track_index = tracks.len();
                tracks.push(FeatureTrack {
                    observations: vec![source, target],
                });
                membership.insert(source, track_index);
                membership.insert(target, track_index);
            }
        }
    }

    (tracks, membership)
}

fn registration_candidates(
    keyframes: &[usize],
    tracks: &[FeatureTrack],
    membership: &HashMap<Observation, usize>,
    seed_pair_index: usize,
    seed_landmarks: &[SeedLandmark],
) -> Vec<RegistrationCandidate> {
    let seed_track_indices: Vec<(usize, usize)> = seed_landmarks
        .iter()
        .filter_map(|seed_landmark| {
            membership
                .get(&Observation {
                    frame_index: seed_pair_index,
                    feature_index: seed_landmark.source_feature_index,
                })
                .map(|&track_index| (seed_landmark.point_index, track_index))
        })
        .collect();

    keyframes
        .iter()
        .copied()
        .filter(|&frame_index| frame_index != seed_pair_index && frame_index != seed_pair_index + 1)
        .map(|frame_index| {
            let correspondences = seed_track_indices
                .iter()
                .filter_map(|&(seed_point_index, track_index)| {
                    tracks[track_index]
                        .observations
                        .iter()
                        .find(|observation| observation.frame_index == frame_index)
                        .map(|observation| RegistrationCorrespondence {
                            seed_point_index,
                            feature_index: observation.feature_index,
                        })
                })
                .collect();
            RegistrationCandidate {
                frame_index,
                correspondences,
            }
        })
        .collect()
}

fn triangulate_track(
    observations: &[RegisteredObservation<'_>],
    seed_frames: &[usize; 2],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<TriangulatedTrack> {
    let mut best: Option<TriangulatedTrack> = None;

    for left_index in 0..observations.len() - 1 {
        for right_index in left_index + 1..observations.len() {
            let left = observations[left_index];
            let right = observations[right_index];
            if seed_frames.contains(&left.frame_index) && seed_frames.contains(&right.frame_index) {
                continue;
            }

            let Some(position) = triangulate_pair(&left, &right, width, height, focal_pixels)
            else {
                continue;
            };
            let Some(triangulation_angle_degrees) =
                triangulation_angle(&position, left.camera, right.camera)
            else {
                continue;
            };
            if triangulation_angle_degrees < MIN_NEW_LANDMARK_TRIANGULATION_ANGLE_DEGREES {
                continue;
            }

            let mut supporting_indices = Vec::new();
            let mut reprojection_errors = Vec::new();
            for (index, observation) in observations.iter().enumerate() {
                let camera_point =
                    observation.camera.rotation * position + observation.camera.translation;
                if camera_point.z <= 1.0e-6 {
                    continue;
                }
                let error = reprojection_error_pixels(
                    &camera_point,
                    observation,
                    width,
                    height,
                    focal_pixels,
                );
                if error.is_finite() && error <= MAX_NEW_LANDMARK_REPROJECTION_ERROR_PIXELS {
                    supporting_indices.push(index);
                    reprojection_errors.push(error);
                }
            }

            if !supporting_indices.contains(&left_index)
                || !supporting_indices.contains(&right_index)
            {
                continue;
            }
            let support_ratio = supporting_indices.len() as f64 / observations.len() as f64;
            if supporting_indices.len() < 2 || support_ratio < MIN_NEW_LANDMARK_SUPPORT_RATIO {
                continue;
            }
            let median_reprojection_error_pixels = median_f64(&mut reprojection_errors)?;
            if median_reprojection_error_pixels > MAX_NEW_LANDMARK_MEDIAN_REPROJECTION_ERROR_PIXELS
            {
                continue;
            }

            let source = supporting_indices
                .iter()
                .map(|&index| observations[index])
                .min_by_key(|observation| observation.frame_index)?;
            let candidate = TriangulatedTrack {
                position,
                source_frame_index: source.frame_index,
                source_feature_index: source.feature_index,
                supporting_observations: supporting_indices.len(),
                median_reprojection_error_pixels,
                triangulation_angle_degrees,
            };
            let replace = best.as_ref().is_none_or(|current| {
                candidate.supporting_observations > current.supporting_observations
                    || (candidate.supporting_observations == current.supporting_observations
                        && (candidate.median_reprojection_error_pixels
                            < current.median_reprojection_error_pixels
                            || (candidate.median_reprojection_error_pixels
                                == current.median_reprojection_error_pixels
                                && candidate.triangulation_angle_degrees
                                    > current.triangulation_angle_degrees)))
            });
            if replace {
                best = Some(candidate);
            }
        }
    }

    best
}

fn triangulate_pair(
    left: &RegisteredObservation<'_>,
    right: &RegisteredObservation<'_>,
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<Vector3<f64>> {
    let center_x = width as f64 * 0.5;
    let center_y = height as f64 * 0.5;
    let x1 = (left.x_pixels - center_x) / focal_pixels;
    let y1 = (left.y_pixels - center_y) / focal_pixels;
    let x2 = (right.x_pixels - center_x) / focal_pixels;
    let y2 = (right.y_pixels - center_y) / focal_pixels;
    let mut system = Matrix4::<f64>::zeros();

    for column in 0..4 {
        system[(0, column)] = x1 * projection_value(left.camera, 2, column)
            - projection_value(left.camera, 0, column);
        system[(1, column)] = y1 * projection_value(left.camera, 2, column)
            - projection_value(left.camera, 1, column);
        system[(2, column)] = x2 * projection_value(right.camera, 2, column)
            - projection_value(right.camera, 0, column);
        system[(3, column)] = y2 * projection_value(right.camera, 2, column)
            - projection_value(right.camera, 1, column);
    }

    let svd = system.svd(false, true);
    let v_t = svd.v_t?;
    let homogeneous = v_t.row(3);
    let w = homogeneous[3];
    if !w.is_finite() || w.abs() <= 1.0e-12 {
        return None;
    }
    let position = Vector3::new(homogeneous[0] / w, homogeneous[1] / w, homogeneous[2] / w);
    position
        .iter()
        .all(|value| value.is_finite())
        .then_some(position)
}

fn projection_value(camera: &RegisteredCamera, row: usize, column: usize) -> f64 {
    if column < 3 {
        camera.rotation[(row, column)]
    } else {
        camera.translation[row]
    }
}

fn triangulation_angle(
    position: &Vector3<f64>,
    left: &RegisteredCamera,
    right: &RegisteredCamera,
) -> Option<f64> {
    let left_ray = *position - left.camera_center();
    let right_ray = *position - right.camera_center();
    if left_ray.norm_squared() <= 1.0e-12 || right_ray.norm_squared() <= 1.0e-12 {
        return None;
    }
    let angle = left_ray
        .normalize()
        .dot(&right_ray.normalize())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    angle.is_finite().then_some(angle)
}

fn reprojection_error_pixels(
    camera_point: &Vector3<f64>,
    observation: &RegisteredObservation<'_>,
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> f64 {
    let projected_x = camera_point.x / camera_point.z * focal_pixels + width as f64 * 0.5;
    let projected_y = camera_point.y / camera_point.z * focal_pixels + height as f64 * 0.5;
    (projected_x - observation.x_pixels).hypot(projected_y - observation.y_pixels)
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

fn select_keyframes(pairs: &[PairStats]) -> Vec<usize> {
    let Some(first_pair) = pairs.first() else {
        return Vec::new();
    };

    let mut keyframes = vec![first_pair.from_frame];
    let mut accumulated_parallax = 0.0f32;
    let mut strongest_pair: Option<(usize, f32)> = None;

    for pair in pairs {
        if pair.matches < 8 || pair.overlap_ratio < MIN_KEYFRAME_OVERLAP {
            accumulated_parallax = 0.0;
            continue;
        }
        accumulated_parallax += pair.median_parallax_residual.max(0.0);
        if !pair.low_parallax && pair.median_parallax_residual >= MIN_STRONG_PAIR_PARALLAX {
            let score = pair.median_parallax_residual * pair.overlap_ratio;
            if strongest_pair.is_none_or(|(_, best_score)| score > best_score) {
                strongest_pair = Some((pair.to_frame, score));
            }
        }

        if accumulated_parallax >= KEYFRAME_PARALLAX_BUDGET {
            if keyframes.last().copied() != Some(pair.to_frame) {
                keyframes.push(pair.to_frame);
            }
            accumulated_parallax = 0.0;
        }
    }

    if keyframes.len() == 1 {
        if let Some((frame_index, _)) = strongest_pair {
            keyframes.push(frame_index);
        }
    }

    keyframes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature_match(a: usize, b: usize) -> FeatureMatch {
        FeatureMatch {
            a,
            b,
            distance: 0.0,
        }
    }

    fn seed_landmarks(count: usize) -> Vec<SeedLandmark> {
        (0..count)
            .map(|index| SeedLandmark {
                point_index: index,
                source_feature_index: index,
            })
            .collect()
    }

    fn pair(
        from_frame: usize,
        overlap_ratio: f32,
        median_parallax_residual: f32,
        low_parallax: bool,
    ) -> PairStats {
        PairStats {
            from_frame,
            to_frame: from_frame + 1,
            features_from: 20,
            features_to: 20,
            matches: 12,
            overlap_ratio,
            median_dx: 0.0,
            median_dy: 0.0,
            median_motion: 2.0,
            median_parallax_residual,
            low_parallax,
        }
    }

    fn camera(frame_index: usize, center_x: f64) -> RegisteredCamera {
        RegisteredCamera {
            frame_index,
            rotation: Matrix3::identity(),
            translation: Vector3::new(-center_x, 0.0, 0.0),
        }
    }

    fn project_feature(
        point: Vector3<f64>,
        camera: &RegisteredCamera,
        width: u32,
        height: u32,
        focal: f64,
    ) -> Feature {
        let camera_point = camera.rotation * point + camera.translation;
        Feature {
            x: (camera_point.x / camera_point.z * focal + width as f64 * 0.5).round() as u32,
            y: (camera_point.y / camera_point.z * focal + height as f64 * 0.5).round() as u32,
            score: 1.0,
            descriptor: vec![0],
        }
    }

    #[test]
    fn chains_adjacent_matches_into_multi_frame_tracks() {
        let matches = vec![
            vec![
                feature_match(0, 0),
                feature_match(1, 1),
                feature_match(2, 2),
            ],
            vec![
                feature_match(0, 0),
                feature_match(1, 1),
                feature_match(3, 3),
            ],
        ];
        let (tracks, _) = build_feature_tracks(&matches);
        let mut lengths: Vec<usize> = tracks
            .iter()
            .map(|track| track.observations.len())
            .collect();
        lengths.sort_unstable();

        assert_eq!(lengths, vec![2, 2, 3, 3]);
    }

    #[test]
    fn reports_registration_ready_keyframes_from_seed_landmark_tracks() {
        let seed_matches: Vec<FeatureMatch> =
            (0..10).map(|index| feature_match(index, index)).collect();
        let next_matches: Vec<FeatureMatch> =
            (0..9).map(|index| feature_match(index, index)).collect();
        let final_matches: Vec<FeatureMatch> =
            (0..8).map(|index| feature_match(index, index)).collect();
        let matches = vec![seed_matches, next_matches, final_matches];
        let (tracks, membership) = build_feature_tracks(&matches);
        let seeds = seed_landmarks(10);

        let candidates = registration_candidates(&[0, 2, 3], &tracks, &membership, 0, &seeds);

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].frame_index, 2);
        assert_eq!(candidates[0].correspondences.len(), 9);
        assert_eq!(candidates[0].correspondences[4].seed_point_index, 4);
        assert_eq!(candidates[0].correspondences[4].feature_index, 4);
        assert_eq!(candidates[1].frame_index, 3);
        assert_eq!(candidates[1].correspondences.len(), 8);
    }

    #[test]
    fn does_not_mark_sparse_track_overlap_as_pnp_ready() {
        let seed_matches: Vec<FeatureMatch> =
            (0..8).map(|index| feature_match(index, index)).collect();
        let next_matches: Vec<FeatureMatch> =
            (0..5).map(|index| feature_match(index, index)).collect();
        let matches = vec![seed_matches, next_matches];
        let analysis = analyze(
            &[pair(0, 0.7, 0.7, false), pair(1, 0.7, 0.7, false)],
            &matches,
            Some((0, &seed_landmarks(8))),
        );

        assert_eq!(analysis.stats.registration_candidates.len(), 1);
        assert_eq!(
            analysis.stats.registration_candidates[0].seed_landmark_correspondences,
            5
        );
        assert!(!analysis.stats.registration_candidates[0].pnp_ready);
    }

    #[test]
    fn triangulates_non_seed_track_from_registered_view() {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let cameras = vec![camera(0, 0.0), camera(1, 0.9), camera(2, 1.7)];
        let point = Vector3::new(0.25, -0.12, 4.2);
        let features: Vec<Vec<Feature>> = cameras
            .iter()
            .map(|camera| vec![project_feature(point, camera, width, height, focal)])
            .collect();
        let matches = vec![vec![feature_match(0, 0)], vec![feature_match(0, 0)]];
        let analysis = analyze(
            &[pair(0, 0.7, 0.7, false), pair(1, 0.7, 0.7, false)],
            &matches,
            Some((0, &[])),
        );

        let result =
            triangulate_new_landmarks(&analysis, &cameras, &features, width, height, focal);

        assert_eq!(result.stats.candidate_tracks, 1);
        assert_eq!(result.stats.accepted_landmarks, 1);
        assert_eq!(result.stats.supporting_observations, 3);
        assert!((result.landmarks[0].position - point).norm() < 0.03);
        assert!(
            result
                .stats
                .median_reprojection_error_pixels
                .unwrap_or(f32::INFINITY)
                < 1.0
        );
        assert!(
            result
                .stats
                .median_triangulation_angle_degrees
                .unwrap_or_default()
                > 0.5
        );
    }

    #[test]
    fn rejects_triangulation_when_non_seed_pair_view_is_behind_landmark() {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let rotation = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let camera_center = Vector3::new(1.0, 0.0, 0.0);
        let cameras = [
            camera(0, 0.0),
            camera(1, 0.5),
            RegisteredCamera {
                frame_index: 2,
                rotation,
                translation: -(rotation * camera_center),
            },
        ];
        let point = Vector3::new(0.0, 0.0, 4.0);
        let projected: Vec<Feature> = cameras
            .iter()
            .map(|camera| project_feature(point, camera, width, height, focal))
            .collect();
        let observations: Vec<RegisteredObservation<'_>> = cameras
            .iter()
            .enumerate()
            .map(|(index, camera)| RegisteredObservation {
                camera,
                frame_index: index,
                feature_index: 0,
                x_pixels: projected[index].x as f64,
                y_pixels: projected[index].y as f64,
            })
            .collect();

        assert!(triangulate_track(&observations, &[0, 1], width, height, focal).is_none());
    }

    #[test]
    fn skips_degenerate_pair_and_uses_later_supported_pair() {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let cameras = [
            camera(0, 0.0),
            camera(1, 0.5),
            RegisteredCamera {
                frame_index: 2,
                rotation: Matrix3::identity(),
                translation: Vector3::new(0.2, 0.0, 1.0),
            },
            camera(3, 1.0),
        ];
        let point = Vector3::new(0.0, 0.0, 4.0);
        let projected: Vec<Feature> = cameras
            .iter()
            .map(|camera| project_feature(point, camera, width, height, focal))
            .collect();
        let observations = [
            RegisteredObservation {
                camera: &cameras[0],
                frame_index: 0,
                feature_index: 0,
                x_pixels: projected[0].x as f64,
                y_pixels: projected[0].y as f64,
            },
            RegisteredObservation {
                camera: &cameras[1],
                frame_index: 1,
                feature_index: 0,
                x_pixels: projected[1].x as f64,
                y_pixels: projected[1].y as f64,
            },
            RegisteredObservation {
                camera: &cameras[2],
                frame_index: 2,
                feature_index: 0,
                x_pixels: 420.0,
                y_pixels: 240.0,
            },
            RegisteredObservation {
                camera: &cameras[3],
                frame_index: 3,
                feature_index: 0,
                x_pixels: projected[3].x as f64,
                y_pixels: projected[3].y as f64,
            },
        ];

        let result = triangulate_track(&observations, &[0, 1], width, height, focal)
            .expect("later supported pair should recover the track");

        assert_eq!(result.supporting_observations, 3);
        assert!((result.position - point).norm() < 0.05);
    }

    #[test]
    fn rejects_new_landmark_with_insufficient_triangulation_angle() {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let cameras = vec![camera(0, 0.0), camera(1, 1.0), camera(2, 2.0)];
        let point = Vector3::new(0.2, 0.1, 250.0);
        let features: Vec<Vec<Feature>> = cameras
            .iter()
            .map(|camera| vec![project_feature(point, camera, width, height, focal)])
            .collect();
        let matches = vec![vec![feature_match(0, 0)], vec![feature_match(0, 0)]];
        let analysis = analyze(
            &[pair(0, 0.7, 0.7, false), pair(1, 0.7, 0.7, false)],
            &matches,
            Some((0, &[])),
        );

        let result =
            triangulate_new_landmarks(&analysis, &cameras, &features, width, height, focal);

        assert_eq!(result.stats.candidate_tracks, 1);
        assert_eq!(result.stats.accepted_landmarks, 0);
        assert!(result.landmarks.is_empty());
    }

    #[test]
    fn selects_keyframes_from_overlap_and_accumulated_parallax() {
        let pairs = vec![
            pair(0, 0.65, 0.45, true),
            pair(1, 0.62, 0.46, true),
            pair(2, 0.58, 0.50, true),
            pair(3, 0.55, 0.70, false),
        ];

        assert_eq!(select_keyframes(&pairs), vec![0, 3]);
    }

    #[test]
    fn weak_or_stationary_links_do_not_create_keyframes() {
        let pairs = vec![
            pair(0, 0.70, 0.0, true),
            pair(1, 0.10, 0.8, false),
            pair(2, 0.75, 0.0, true),
        ];

        assert_eq!(select_keyframes(&pairs), vec![0]);
    }
}
