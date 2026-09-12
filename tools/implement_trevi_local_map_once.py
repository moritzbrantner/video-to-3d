#!/usr/bin/env python3
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement target, found {count}")
    path.write_text(text.replace(old, new, 1))


multi = Path("crates/video-to-3d-core/src/multi_view.rs")
replace_once(
    multi,
    '''#[derive(Clone, Debug)]
pub(super) struct NewLandmarkAnalysis {
    pub stats: NewLandmarkStats,
    pub landmarks: Vec<NewLandmark>,
}
''',
    '''#[derive(Clone, Debug)]
pub(super) struct NewLandmarkAnalysis {
    pub stats: NewLandmarkStats,
    pub landmarks: Vec<NewLandmark>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LocalMapRegistrationCorrespondence {
    pub point: Vector3<f64>,
    pub feature_index: usize,
}

#[derive(Clone, Debug)]
pub(super) struct LocalMapRegistrationCandidate {
    pub frame_index: usize,
    pub correspondences: Vec<LocalMapRegistrationCorrespondence>,
}
''',
)

replace_once(
    multi,
    '''    NewLandmarkAnalysis { stats, landmarks }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bundle_adjust(
''',
    '''    NewLandmarkAnalysis { stats, landmarks }
}

pub(super) fn local_map_registration_candidates(
    analysis: &MultiViewAnalysis,
    seed_points: &[Vector3<f64>],
    new_landmarks: &[NewLandmark],
    registered_frames: &HashSet<usize>,
) -> Vec<LocalMapRegistrationCandidate> {
    let mut map_landmarks: Vec<(usize, Vector3<f64>)> = analysis
        .seed_tracks
        .iter()
        .filter_map(|seed| {
            seed_points
                .get(seed.point_index)
                .copied()
                .map(|position| (seed.track_index, position))
        })
        .collect();
    map_landmarks.extend(
        new_landmarks
            .iter()
            .map(|landmark| (landmark.track_index, landmark.position)),
    );
    map_landmarks.sort_by_key(|landmark| landmark.0);
    map_landmarks.dedup_by_key(|landmark| landmark.0);

    let mut candidates: Vec<LocalMapRegistrationCandidate> = analysis
        .stats
        .keyframes
        .iter()
        .copied()
        .filter(|frame_index| !registered_frames.contains(frame_index))
        .map(|frame_index| {
            let correspondences = map_landmarks
                .iter()
                .filter_map(|(track_index, position)| {
                    analysis
                        .tracks
                        .get(*track_index)?
                        .observations
                        .iter()
                        .find(|observation| observation.frame_index == frame_index)
                        .map(|observation| LocalMapRegistrationCorrespondence {
                            point: *position,
                            feature_index: observation.feature_index,
                        })
                })
                .collect();
            LocalMapRegistrationCandidate {
                frame_index,
                correspondences,
            }
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .correspondences
            .len()
            .cmp(&left.correspondences.len())
            .then_with(|| left.frame_index.cmp(&right.frame_index))
    });
    candidates
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bundle_adjust(
''',
)

test = r'''

    #[test]
    fn local_map_correspondences_extend_beyond_seed_track_support() {
        let matches: Vec<Vec<FeatureMatch>> = (0..3)
            .map(|_| (0..10).map(|index| feature_match(index, index)).collect())
            .collect();
        let mut analysis = analyze(
            &[
                pair(0, 0.7, 1.3, false),
                pair(1, 0.7, 1.3, false),
                pair(2, 0.7, 1.3, false),
            ],
            &matches,
            Some((0, &seed_landmarks(1))),
        );
        analysis.stats.keyframes = vec![3];

        let mut new_landmarks: Vec<NewLandmark> = (1..9)
            .map(|track_index| NewLandmark {
                position: Vector3::new(track_index as f64 * 0.1, 0.0, 4.0),
                source_frame_index: 1,
                source_feature_index: track_index,
                supporting_observations: 2,
                median_reprojection_error_pixels: 0.1,
                triangulation_angle_degrees: 1.0,
                track_index,
            })
            .collect();
        new_landmarks.push(new_landmarks[0].clone());
        let registered_frames: HashSet<usize> = [0, 1, 2].into_iter().collect();
        let seed_points = [Vector3::new(0.0, 0.0, 4.0)];

        let candidates = local_map_registration_candidates(
            &analysis,
            &seed_points,
            &new_landmarks,
            &registered_frames,
        );
        let frame_three = candidates
            .iter()
            .find(|candidate| candidate.frame_index == 3)
            .expect("frame 3 local-map candidate");

        assert_eq!(frame_three.correspondences.len(), 9);
        assert_eq!(frame_three.correspondences[0].feature_index, 0);
    }
'''
text = multi.read_text()
closing = text.rfind("\n}")
if closing < 0:
    raise SystemExit("multi_view.rs: test module closing brace not found")
multi.write_text(text[:closing] + test + text[closing:])

reconstruction = Path("crates/video-to-3d-core/src/reconstruction.rs")
replace_once(
    reconstruction,
    '''const REGISTRATION_RECOVERY_MIN_FEATURE_DISTANCE: u32 = 5;
const MIN_TRACK_MATCHES: usize = 8;
''',
    '''const REGISTRATION_RECOVERY_MIN_FEATURE_DISTANCE: u32 = 5;
const MAX_INCREMENTAL_REGISTRATION_ROUNDS: usize = 3;
const MIN_TRACK_MATCHES: usize = 8;
''',
)

replace_once(
    reconstruction,
    '''    registered_views.sort_by_key(|view| view.frame_index);
    registered_cameras.sort_by_key(|camera| camera.frame_index);
    registered_geometry.sort_by_key(|camera| camera.frame_index);

    let new_landmark_analysis = multi_view::triangulate_new_landmarks(
''',
    '''    registered_views.sort_by_key(|view| view.frame_index);
    registered_cameras.sort_by_key(|camera| camera.frame_index);
    registered_geometry.sort_by_key(|camera| camera.frame_index);

    if let Some((_, estimate)) = best_two_view.as_ref() {
        let seed_points: Vec<nalgebra::Vector3<f64>> = estimate
            .points
            .iter()
            .map(|point| point.position)
            .collect();

        for _ in 0..MAX_INCREMENTAL_REGISTRATION_ROUNDS {
            let local_map = multi_view::triangulate_new_landmarks(
                &multi_view_analysis,
                &registered_geometry,
                &features,
                width,
                height,
                focal as f64,
            );
            if local_map.landmarks.is_empty() {
                break;
            }

            let registered_frames: HashSet<usize> = registered_geometry
                .iter()
                .map(|camera| camera.frame_index)
                .collect();
            let candidates = multi_view::local_map_registration_candidates(
                &multi_view_analysis,
                &seed_points,
                &local_map.landmarks,
                &registered_frames,
            );
            let mut accepted_this_round = 0usize;

            for candidate in candidates {
                if candidate.correspondences.len() < 8 {
                    continue;
                }
                let pnp_correspondences: Vec<pnp::PnpCorrespondence> = candidate
                    .correspondences
                    .iter()
                    .filter_map(|correspondence| {
                        let feature = features
                            .get(candidate.frame_index)?
                            .get(correspondence.feature_index)?;
                        Some(pnp::PnpCorrespondence {
                            point: correspondence.point,
                            x_pixels: feature.x as f64,
                            y_pixels: feature.y as f64,
                        })
                    })
                    .collect();
                if pnp_correspondences.len() != candidate.correspondences.len() {
                    continue;
                }
                let Some(pose) =
                    pnp::estimate_pose(&pnp_correspondences, width, height, focal as f64)
                else {
                    continue;
                };

                registered_geometry.push(multi_view::RegisteredCamera {
                    frame_index: candidate.frame_index,
                    rotation: pose.rotation,
                    translation: pose.translation,
                });
                registered_cameras.push(CameraPose {
                    frame_index: candidate.frame_index,
                    x: pose.camera_center.x as f32,
                    y: pose.camera_center.y as f32,
                    z: pose.camera_center.z as f32,
                    matched_features: pose.inliers,
                });
                registered_views.push(RegisteredViewStats {
                    frame_index: candidate.frame_index,
                    correspondences: pnp_correspondences.len(),
                    inliers: pose.inliers,
                    inlier_ratio: pose.inliers as f32 / pnp_correspondences.len() as f32,
                    median_reprojection_error_pixels: pose.median_reprojection_error_pixels as f32,
                    rotation: [
                        pose.rotation[(0, 0)] as f32,
                        pose.rotation[(0, 1)] as f32,
                        pose.rotation[(0, 2)] as f32,
                        pose.rotation[(1, 0)] as f32,
                        pose.rotation[(1, 1)] as f32,
                        pose.rotation[(1, 2)] as f32,
                        pose.rotation[(2, 0)] as f32,
                        pose.rotation[(2, 1)] as f32,
                        pose.rotation[(2, 2)] as f32,
                    ],
                    translation: [
                        pose.translation.x as f32,
                        pose.translation.y as f32,
                        pose.translation.z as f32,
                    ],
                    recovered_from_revisit: false,
                });
                accepted_this_round += 1;
            }

            if accepted_this_round == 0 {
                break;
            }
            registered_views.sort_by_key(|view| view.frame_index);
            registered_cameras.sort_by_key(|camera| camera.frame_index);
            registered_geometry.sort_by_key(|camera| camera.frame_index);
        }
    }

    let new_landmark_analysis = multi_view::triangulate_new_landmarks(
''',
)
