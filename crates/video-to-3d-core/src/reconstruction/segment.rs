//! Seeding, registration, triangulation and bundle adjustment of one clip segment.

use super::control::ReconstructionControl;
use super::seed::{calibrated_pair_stats, seed_pair_cameras};
use super::types::{CameraPose, Feature, FeatureMatch, RegisteredViewStats, SegmentSolveStats};
use crate::{
    keyframe_selection, multi_view, pnp, revisit, two_view, ClipSegmentStats, RevisitStats,
};
use std::collections::HashSet;

/// Rounds of incremental registration against triangulated landmarks (#230).
const MAX_EXPANSION_ROUNDS: usize = 64;
/// Known-landmark correspondences a frame needs before an expansion PnP attempt.
const MIN_EXPANSION_CORRESPONDENCES: usize = 12;

/// Seed, register, triangulate and bundle-adjust one clip segment in its own frame.
pub(super) struct SegmentSolve {
    pub(super) multi_view_analysis: multi_view::MultiViewAnalysis,
    pub(super) revisits: RevisitStats,
    pub(super) registered_views: Vec<RegisteredViewStats>,
    pub(super) registered_cameras: Vec<CameraPose>,
    pub(super) registered_geometry: Vec<multi_view::RegisteredCamera>,
    pub(super) new_landmarks: Vec<multi_view::NewLandmark>,
    pub(super) optimized_seed_points: Vec<nalgebra::Vector3<f64>>,
    pub(super) optimized_new_landmark_positions: Vec<nalgebra::Vector3<f64>>,
    /// Views registered by incremental expansion rather than the seed's tracks.
    pub(super) expanded_views: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve_segment(
    seed: Option<&([usize; 2], two_view::TwoViewEstimate)>,
    keyframe_selection: &keyframe_selection::KeyframeSelection,
    adjacent_matches: &[Vec<FeatureMatch>],
    features: &[Vec<Feature>],
    revisit_context: &revisit::RevisitContext<'_>,
    width: u32,
    height: u32,
    focal: f64,
    secondary: bool,
    control: &mut ReconstructionControl<'_>,
) -> Result<SegmentSolve, String> {
    let seed_landmarks = seed.map(|(seed_frames, estimate)| {
        (
            *seed_frames,
            estimate
                .points
                .iter()
                .enumerate()
                .map(|(point_index, point)| multi_view::SeedLandmark {
                    point_index,
                    source_feature_index: point.source_feature_index,
                })
                .collect::<Vec<_>>(),
        )
    });
    // Tracks extend only through adjacent matches that agree with that pair's
    // essential geometry, so registration does not inherit descriptor mismatches.
    let verified_matches: Vec<Vec<FeatureMatch>> = adjacent_matches
        .iter()
        .enumerate()
        .map(|(pair, matches)| {
            let points: Vec<two_view::PixelCorrespondence> = matches
                .iter()
                .filter_map(|feature_match| {
                    let a = features.get(pair)?.get(feature_match.a)?;
                    let b = features.get(pair + 1)?.get(feature_match.b)?;
                    Some(two_view::PixelCorrespondence {
                        source_feature_index: feature_match.a,
                        descriptor_distance: feature_match.distance,
                        source: [a.x as f64, a.y as f64],
                        target: [b.x as f64, b.y as f64],
                    })
                })
                .collect();
            match (points.len() == matches.len())
                .then(|| two_view::essential_inlier_mask(&points, width, height, focal))
                .flatten()
            {
                // A pair whose matches already agree almost entirely keeps them all, so
                // threshold noise never trims consistent synthetic or slow-motion tracks.
                Some(mask) if mask.iter().filter(|&&inlier| !inlier).count() * 10 > mask.len() => {
                    matches
                        .iter()
                        .zip(mask)
                        .filter_map(|(feature_match, inlier)| inlier.then_some(*feature_match))
                        .collect()
                }
                _ => matches.clone(),
            }
        })
        .collect();
    let adjacent_matches = verified_matches.as_slice();
    let mut multi_view_analysis = multi_view::analyze(
        keyframe_selection,
        adjacent_matches,
        seed_landmarks
            .as_ref()
            .map(|(seed_frames, landmarks)| (*seed_frames, landmarks.as_slice())),
    );
    let seed_segment = seed
        .map(|(seed_frames, _)| keyframe_selection.segment_keyframes(seed_frames[0]))
        .unwrap_or_default();
    // The primary solve screens the whole clip for its diagnostics; a secondary solve only
    // screens its own segment, since only that segment can recover or close its cameras.
    let revisit_keyframes = if secondary {
        &seed_segment
    } else {
        &multi_view_analysis.stats.keyframes
    };
    let mut revisits = revisit::analyze(
        revisit_context,
        revisit_keyframes,
        seed.map(|(seed_frames, _)| *seed_frames),
        &seed_segment,
    );

    control.check()?;
    let mut registered_views = Vec::new();
    let mut registered_cameras = Vec::new();
    let mut registered_geometry = Vec::new();
    if let Some((seed_frames, estimate)) = seed {
        registered_geometry.push(multi_view::RegisteredCamera {
            frame_index: seed_frames[0],
            rotation: nalgebra::Matrix3::identity(),
            translation: nalgebra::Vector3::zeros(),
        });
        registered_geometry.push(multi_view::RegisteredCamera {
            frame_index: seed_frames[1],
            rotation: estimate.rotation,
            translation: estimate.translation,
        });

        for candidate in &multi_view_analysis.registration_candidates {
            control.check()?;
            if candidate.correspondences.len() < 8 {
                continue;
            }
            let pnp_correspondences: Vec<pnp::PnpCorrespondence> = candidate
                .correspondences
                .iter()
                .filter_map(|correspondence| {
                    let point = estimate.points.get(correspondence.seed_point_index)?;
                    let feature = features
                        .get(candidate.frame_index)?
                        .get(correspondence.feature_index)?;
                    Some(pnp::PnpCorrespondence {
                        point: point.position,
                        x_pixels: feature.x as f64,
                        y_pixels: feature.y as f64,
                    })
                })
                .collect();
            if pnp_correspondences.len() != candidate.correspondences.len() {
                continue;
            }
            let Some(pose) = pnp::estimate_pose_with_control(
                &pnp_correspondences,
                width,
                height,
                focal,
                control,
            ) else {
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
        }

        let registered_frames: HashSet<usize> = registered_geometry
            .iter()
            .map(|camera| camera.frame_index)
            .collect();
        let candidate_frames: Vec<usize> = multi_view_analysis
            .registration_candidates
            .iter()
            .map(|candidate| candidate.frame_index)
            .collect();
        for recovered in revisit::recover_failed_registrations(
            &mut revisits,
            *seed_frames,
            estimate,
            &candidate_frames,
            &registered_frames,
            revisit_context,
        ) {
            registered_geometry.push(multi_view::RegisteredCamera {
                frame_index: recovered.frame_index,
                rotation: recovered.rotation,
                translation: recovered.translation,
            });
            registered_cameras.push(CameraPose {
                frame_index: recovered.frame_index,
                x: recovered.camera_center.x as f32,
                y: recovered.camera_center.y as f32,
                z: recovered.camera_center.z as f32,
                matched_features: recovered.inliers,
            });
            registered_views.push(RegisteredViewStats {
                frame_index: recovered.frame_index,
                correspondences: recovered.correspondences,
                inliers: recovered.inliers,
                inlier_ratio: recovered.inliers as f32 / recovered.correspondences as f32,
                median_reprojection_error_pixels: recovered.median_reprojection_error_pixels as f32,
                rotation: [
                    recovered.rotation[(0, 0)] as f32,
                    recovered.rotation[(0, 1)] as f32,
                    recovered.rotation[(0, 2)] as f32,
                    recovered.rotation[(1, 0)] as f32,
                    recovered.rotation[(1, 1)] as f32,
                    recovered.rotation[(1, 2)] as f32,
                    recovered.rotation[(2, 0)] as f32,
                    recovered.rotation[(2, 1)] as f32,
                    recovered.rotation[(2, 2)] as f32,
                ],
                translation: [
                    recovered.translation.x as f32,
                    recovered.translation.y as f32,
                    recovered.translation.z as f32,
                ],
                recovered_from_revisit: true,
            });
        }
    }
    let mut expanded_views = 0;
    // Incremental expansion (#230): frames the seed's tracks cannot reach register
    // against landmarks triangulated from the cameras registered so far, round by
    // round, each pose through the same PnP gates. Stops when a round adds nothing.
    if let Some((seed_frames, estimate)) = seed {
        let seed_positions: Vec<nalgebra::Vector3<f64>> =
            estimate.points.iter().map(|point| point.position).collect();
        let segment_frames = keyframe_selection.segment_keyframes(seed_frames[0]);
        for _round in 0..MAX_EXPANSION_ROUNDS {
            control.check()?;
            let landmarks = multi_view::triangulate_new_landmarks(
                &multi_view_analysis,
                &registered_geometry,
                features,
                width,
                height,
                focal,
            )
            .landmarks;
            let registered: HashSet<usize> = registered_geometry
                .iter()
                .map(|camera| camera.frame_index)
                .collect();
            let mut added = 0;
            for &frame_index in &segment_frames {
                if registered.contains(&frame_index) {
                    continue;
                }
                let correspondences: Vec<pnp::PnpCorrespondence> = multi_view_analysis
                    .landmark_correspondences(&seed_positions, &landmarks, frame_index)
                    .into_iter()
                    .filter_map(|(position, feature_index)| {
                        let feature = features.get(frame_index)?.get(feature_index)?;
                        Some(pnp::PnpCorrespondence {
                            point: position,
                            x_pixels: feature.x as f64,
                            y_pixels: feature.y as f64,
                        })
                    })
                    .collect();
                if correspondences.len() < MIN_EXPANSION_CORRESPONDENCES {
                    continue;
                }
                let Some(pose) = pnp::estimate_pose_with_control(
                    &correspondences,
                    width,
                    height,
                    focal,
                    control,
                ) else {
                    continue;
                };
                registered_geometry.push(multi_view::RegisteredCamera {
                    frame_index,
                    rotation: pose.rotation,
                    translation: pose.translation,
                });
                registered_cameras.push(CameraPose {
                    frame_index,
                    x: pose.camera_center.x as f32,
                    y: pose.camera_center.y as f32,
                    z: pose.camera_center.z as f32,
                    matched_features: pose.inliers,
                });
                registered_views.push(RegisteredViewStats {
                    frame_index,
                    correspondences: correspondences.len(),
                    inliers: pose.inliers,
                    inlier_ratio: pose.inliers as f32 / correspondences.len() as f32,
                    median_reprojection_error_pixels: pose.median_reprojection_error_pixels as f32,
                    rotation: [0, 1, 2]
                        .iter()
                        .flat_map(|&row| (0..3).map(move |column| (row, column)))
                        .map(|(row, column)| pose.rotation[(row, column)] as f32)
                        .collect::<Vec<_>>()
                        .try_into()
                        .expect("nine rotation entries"),
                    translation: [
                        pose.translation.x as f32,
                        pose.translation.y as f32,
                        pose.translation.z as f32,
                    ],
                    recovered_from_revisit: false,
                });
                added += 1;
                expanded_views += 1;
            }
            if added == 0 {
                break;
            }
        }
    }
    registered_views.sort_by_key(|view| view.frame_index);
    registered_cameras.sort_by_key(|camera| camera.frame_index);
    registered_geometry.sort_by_key(|camera| camera.frame_index);

    let new_landmark_analysis = multi_view::triangulate_new_landmarks(
        &multi_view_analysis,
        &registered_geometry,
        features,
        width,
        height,
        focal,
    );
    multi_view_analysis.stats.new_landmarks = new_landmark_analysis.stats.clone();
    let mut new_landmarks = new_landmark_analysis.landmarks;

    let mut optimized_seed_points = seed
        .map(|(_, estimate)| {
            estimate
                .points
                .iter()
                .map(|point| point.position)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut optimized_new_landmark_positions: Vec<nalgebra::Vector3<f64>> = new_landmarks
        .iter()
        .map(|landmark| landmark.position)
        .collect();
    if let Some((seed_frames, estimate)) = seed {
        let adjustment = multi_view::bundle_adjust_with_control(
            &multi_view_analysis,
            &optimized_seed_points,
            &new_landmarks,
            &registered_geometry,
            features,
            width,
            height,
            focal,
            control,
        );
        control.check()?;
        multi_view_analysis.stats.bundle_adjustment = adjustment.stats.clone();
        optimized_seed_points = adjustment.seed_points;
        optimized_new_landmark_positions = adjustment.new_landmark_positions;
        registered_geometry = adjustment.cameras;
        for (landmark, position) in new_landmarks
            .iter_mut()
            .zip(&optimized_new_landmark_positions)
        {
            landmark.position = *position;
        }

        let pre_loop_geometry = registered_geometry.clone();
        let pre_loop_seed_points = optimized_seed_points.clone();
        let pre_loop_new_landmark_positions = optimized_new_landmark_positions.clone();
        let pre_loop_adjustment = multi_view_analysis.stats.bundle_adjustment.clone();
        let closures = revisit::close_registered_drift(
            &mut revisits,
            *seed_frames,
            estimate,
            &registered_geometry,
            revisit_context,
        );

        if !closures.is_empty() {
            for closure in &closures {
                control.check()?;
                if let Some(camera) = registered_geometry
                    .iter_mut()
                    .find(|camera| camera.frame_index == closure.frame_index)
                {
                    camera.rotation = closure.rotation;
                    camera.translation = closure.translation;
                }
            }

            let loop_adjustment = multi_view::bundle_adjust_with_control(
                &multi_view_analysis,
                &optimized_seed_points,
                &new_landmarks,
                &registered_geometry,
                features,
                width,
                height,
                focal,
                control,
            );
            control.check()?;
            let retained = loop_adjustment.stats.accepted
                && revisit::closure_corrections_retained(
                    &closures,
                    &pre_loop_geometry,
                    &loop_adjustment.cameras,
                );

            if retained {
                multi_view_analysis.stats.bundle_adjustment = loop_adjustment.stats;
                optimized_seed_points = loop_adjustment.seed_points;
                optimized_new_landmark_positions = loop_adjustment.new_landmark_positions;
                registered_geometry = loop_adjustment.cameras;
                for (landmark, position) in new_landmarks
                    .iter_mut()
                    .zip(&optimized_new_landmark_positions)
                {
                    landmark.position = *position;
                }
            } else {
                let closure_frames: HashSet<usize> =
                    closures.iter().map(|closure| closure.frame_index).collect();
                for closure in &mut revisits.closures {
                    if closure_frames.contains(&closure.frame_index) {
                        closure.accepted = false;
                    }
                }
                registered_geometry = pre_loop_geometry;
                optimized_seed_points = pre_loop_seed_points;
                optimized_new_landmark_positions = pre_loop_new_landmark_positions;
                multi_view_analysis.stats.bundle_adjustment = pre_loop_adjustment;
                for (landmark, position) in new_landmarks
                    .iter_mut()
                    .zip(&optimized_new_landmark_positions)
                {
                    landmark.position = *position;
                }
            }
        }

        registered_cameras = registered_geometry
            .iter()
            .filter(|camera| !seed_frames.contains(&camera.frame_index))
            .filter_map(|camera| {
                let view = registered_views
                    .iter()
                    .find(|view| view.frame_index == camera.frame_index)?;
                let center = camera.camera_center();
                Some(CameraPose {
                    frame_index: camera.frame_index,
                    x: center.x as f32,
                    y: center.y as f32,
                    z: center.z as f32,
                    matched_features: view.inliers,
                })
            })
            .collect();
    }

    control.check()?;
    Ok(SegmentSolve {
        multi_view_analysis,
        revisits,
        registered_views,
        registered_cameras,
        registered_geometry,
        new_landmarks,
        optimized_seed_points,
        optimized_new_landmark_positions,
        expanded_views,
    })
}

pub(super) fn segment_solve_stats(
    segment: usize,
    span: &ClipSegmentStats,
    primary: bool,
    seed: Option<&([usize; 2], two_view::TwoViewEstimate)>,
    solve: Option<&SegmentSolve>,
    focal: f32,
) -> SegmentSolveStats {
    let stats = solve.map(|solve| &solve.multi_view_analysis.stats);
    let cameras = match (seed, solve) {
        (Some((seed_frames, estimate)), Some(solve)) => {
            let mut cameras = seed_pair_cameras(*seed_frames, estimate);
            cameras.extend(solve.registered_cameras.iter().copied());
            cameras.sort_by_key(|camera| camera.frame_index);
            cameras
        }
        _ => Vec::new(),
    };
    SegmentSolveStats {
        segment,
        first_frame: span.first_frame,
        last_frame: span.last_frame,
        primary,
        seed_pair: seed
            .map(|(seed_frames, estimate)| calibrated_pair_stats(*seed_frames, estimate, focal)),
        registration_candidates: stats.map_or(0, |stats| stats.registration_candidates.len()),
        pnp_ready_candidates: stats.map_or(0, |stats| {
            stats
                .registration_candidates
                .iter()
                .filter(|candidate| candidate.pnp_ready)
                .count()
        }),
        registered_views: solve.map_or(0, |solve| solve.registered_views.len()),
        recovered_from_revisit: solve.map_or(0, |solve| {
            solve
                .registered_views
                .iter()
                .filter(|view| view.recovered_from_revisit)
                .count()
        }),
        sparse_points: seed.map_or(0, |(_, estimate)| estimate.points.len())
            + solve.map_or(0, |solve| solve.new_landmarks.len()),
        bundle_adjustment_accepted: stats.is_some_and(|stats| stats.bundle_adjustment.accepted),
        cameras,
    }
}
