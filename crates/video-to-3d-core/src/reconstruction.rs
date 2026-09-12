mod dense;
mod multi_view;
mod pnp;
mod revisit;
mod two_view;

pub use dense::DenseStats;
pub use multi_view::{
    BundleAdjustmentStats, MultiViewStats, NewLandmarkStats, RegistrationCandidateStats,
};
pub use revisit::{RevisitCandidateStats, RevisitClosureStats, RevisitRecoveryStats, RevisitStats};

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FrameInput {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct ReconstructionOptions {
    pub max_features: usize,
    pub min_feature_distance: u32,
    pub descriptor_radius: u32,
    pub match_radius: u32,
    pub max_descriptor_distance: f32,
    pub ratio_threshold: f32,
    pub focal_length_pixels: Option<f32>,
}

impl Default for ReconstructionOptions {
    fn default() -> Self {
        Self {
            max_features: 320,
            min_feature_distance: 7,
            descriptor_radius: 3,
            match_radius: 42,
            max_descriptor_distance: 36.0,
            ratio_threshold: 0.82,
            focal_length_pixels: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReconstructionRequest {
    pub frames: Vec<FrameInput>,
    #[serde(default)]
    pub options: ReconstructionOptions,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CameraPose {
    pub frame_index: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub matched_features: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PairStats {
    pub from_frame: usize,
    pub to_frame: usize,
    pub features_from: usize,
    pub features_to: usize,
    pub matches: usize,
    pub overlap_ratio: f32,
    pub median_dx: f32,
    pub median_dy: f32,
    pub median_motion: f32,
    pub median_parallax_residual: f32,
    pub low_parallax: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CalibratedPairStats {
    pub from_frame: usize,
    pub to_frame: usize,
    pub matches: usize,
    pub inliers: usize,
    pub inlier_ratio: f32,
    pub focal_pixels: f32,
    pub median_sampson_error_pixels: f32,
    pub median_reprojection_error_pixels: f32,
    pub median_triangulation_angle_degrees: f32,
    pub relative_rotation: [f32; 9],
    pub translation_direction: [f32; 3],
}

#[derive(Clone, Debug, Serialize)]
pub struct RegisteredViewStats {
    pub frame_index: usize,
    pub correspondences: usize,
    pub inliers: usize,
    pub inlier_ratio: f32,
    pub median_reprojection_error_pixels: f32,
    pub rotation: [f32; 9],
    pub translation: [f32; 3],
    pub recovered_from_revisit: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReconstructionResult {
    pub cameras: Vec<CameraPose>,
    pub points: Vec<Point3>,
    pub dense_points: Vec<Point3>,
    pub dense: DenseStats,
    pub pairs: Vec<PairStats>,
    pub calibrated_pair: Option<CalibratedPairStats>,
    pub multi_view: MultiViewStats,
    pub revisits: RevisitStats,
    pub registered_views: Vec<RegisteredViewStats>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
struct Feature {
    x: u32,
    y: u32,
    score: f32,
    descriptor: Vec<i16>,
}

#[derive(Clone, Copy, Debug)]
struct FeatureMatch {
    a: usize,
    b: usize,
    distance: f32,
}

pub fn reconstruct(request: &ReconstructionRequest) -> Result<ReconstructionResult, String> {
    validate_request(request)?;

    let width = request.frames[0].width;
    let height = request.frames[0].height;
    let options = request.options;
    let luma_frames: Vec<Vec<u8>> = request.frames.iter().map(to_luma).collect();
    let features: Vec<Vec<Feature>> = luma_frames
        .iter()
        .map(|luma| detect_features(luma, width, height, options))
        .collect();

    let focal = options
        .focal_length_pixels
        .unwrap_or(0.86 * width.max(height) as f32);
    let diagonal = (width as f32).hypot(height as f32);
    let mut cameras = Vec::with_capacity(request.frames.len());
    let mut points = Vec::new();
    let mut pairs = Vec::with_capacity(request.frames.len() - 1);
    let mut adjacent_matches = Vec::with_capacity(request.frames.len() - 1);
    let mut warnings = Vec::new();
    let mut best_two_view: Option<(usize, two_view::TwoViewEstimate)> = None;

    let mut camera = CameraPose {
        frame_index: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        matched_features: 0,
    };
    cameras.push(camera);

    for pair_index in 0..request.frames.len() - 1 {
        let source_features = &features[pair_index];
        let target_features = &features[pair_index + 1];
        let matches = match_features(source_features, target_features, options);
        let overlap_ratio = if source_features.is_empty() || target_features.is_empty() {
            0.0
        } else {
            matches.len() as f32 / source_features.len().min(target_features.len()) as f32
        };

        let mut dx_values = Vec::with_capacity(matches.len());
        let mut dy_values = Vec::with_capacity(matches.len());
        let mut motion_values = Vec::with_capacity(matches.len());
        for feature_match in &matches {
            let a = &source_features[feature_match.a];
            let b = &target_features[feature_match.b];
            let dx = b.x as f32 - a.x as f32;
            let dy = b.y as f32 - a.y as f32;
            dx_values.push(dx);
            dy_values.push(dy);
            motion_values.push(dx.hypot(dy));
        }

        let median_dx = median(&mut dx_values);
        let median_dy = median(&mut dy_values);
        let median_motion = median(&mut motion_values);
        let (translation_x, translation_y, parallax_residuals) =
            compensate_global_motion(source_features, target_features, &matches, width, height);
        let mut residuals_for_median = parallax_residuals.clone();
        let median_parallax_residual = median(&mut residuals_for_median);
        let low_parallax =
            matches.len() < 6 || median_motion < 1.4 || median_parallax_residual < 0.55;

        if !low_parallax {
            if let Some(estimate) = two_view::estimate_two_view(
                source_features,
                target_features,
                &matches,
                width,
                height,
                focal as f64,
            ) {
                let replace = best_two_view
                    .as_ref()
                    .is_none_or(|(_, current)| estimate.is_better_than(current));
                if replace {
                    best_two_view = Some((pair_index, estimate));
                }
            }
        }

        let observed_baseline = if low_parallax {
            0.0
        } else {
            (median_parallax_residual / diagonal * 12.0).clamp(0.05, 1.0)
        };

        if low_parallax {
            camera = CameraPose {
                frame_index: pair_index + 1,
                matched_features: matches.len(),
                ..camera
            };
        } else {
            camera = CameraPose {
                frame_index: pair_index + 1,
                x: camera.x - translation_x / width as f32 * 1.5,
                y: camera.y + translation_y / height as f32 * 1.5,
                z: camera.z + observed_baseline,
                matched_features: matches.len(),
            };
        }
        cameras.push(camera);

        if !low_parallax {
            for (feature_match, disparity) in matches.iter().zip(&parallax_residuals) {
                if *disparity < 0.35 {
                    continue;
                }
                let a = &source_features[feature_match.a];
                let depth = (focal * observed_baseline / disparity.max(0.5)).clamp(0.35, 18.0);
                let normalized_x = (a.x as f32 - width as f32 * 0.5) / focal;
                let normalized_y = (a.y as f32 - height as f32 * 0.5) / focal;
                let source_camera = cameras[pair_index];
                let (r, g, b) = sample_rgb(&request.frames[pair_index], a.x, a.y);
                let descriptor_confidence = (1.0
                    - feature_match.distance / options.max_descriptor_distance)
                    .clamp(0.0, 1.0);
                let motion_confidence = (disparity / 5.0).clamp(0.15, 1.0);

                points.push(Point3 {
                    x: source_camera.x + normalized_x * depth,
                    y: source_camera.y - normalized_y * depth,
                    z: source_camera.z + depth,
                    confidence: descriptor_confidence * motion_confidence,
                    r,
                    g,
                    b,
                });
            }
        }

        pairs.push(PairStats {
            from_frame: pair_index,
            to_frame: pair_index + 1,
            features_from: source_features.len(),
            features_to: target_features.len(),
            matches: matches.len(),
            overlap_ratio,
            median_dx,
            median_dy,
            median_motion,
            median_parallax_residual,
            low_parallax,
        });
        adjacent_matches.push(matches);
    }

    let seed_landmarks = best_two_view.as_ref().map(|(pair_index, estimate)| {
        (
            *pair_index,
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
    let mut multi_view_analysis = multi_view::analyze(
        &pairs,
        &adjacent_matches,
        seed_landmarks
            .as_ref()
            .map(|(pair_index, landmarks)| (*pair_index, landmarks.as_slice())),
    );
    let revisit_context =
        revisit::RevisitContext::new(&features, width, height, focal as f64, options);
    let mut revisits = revisit::analyze(
        &revisit_context,
        &multi_view_analysis.stats.keyframes,
        best_two_view.as_ref().map(|(pair_index, _)| *pair_index),
    );

    let mut registered_views = Vec::new();
    let mut registered_cameras = Vec::new();
    let mut registered_geometry = Vec::new();
    if let Some((seed_pair_index, estimate)) = best_two_view.as_ref() {
        registered_geometry.push(multi_view::RegisteredCamera {
            frame_index: *seed_pair_index,
            rotation: nalgebra::Matrix3::identity(),
            translation: nalgebra::Vector3::zeros(),
        });
        registered_geometry.push(multi_view::RegisteredCamera {
            frame_index: *seed_pair_index + 1,
            rotation: estimate.rotation,
            translation: estimate.translation,
        });

        for candidate in &multi_view_analysis.registration_candidates {
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
            let Some(pose) = pnp::estimate_pose(&pnp_correspondences, width, height, focal as f64)
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
            *seed_pair_index,
            estimate,
            &candidate_frames,
            &registered_frames,
            &revisit_context,
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
    registered_views.sort_by_key(|view| view.frame_index);
    registered_cameras.sort_by_key(|camera| camera.frame_index);
    registered_geometry.sort_by_key(|camera| camera.frame_index);

    let new_landmark_analysis = multi_view::triangulate_new_landmarks(
        &multi_view_analysis,
        &registered_geometry,
        &features,
        width,
        height,
        focal as f64,
    );
    multi_view_analysis.stats.new_landmarks = new_landmark_analysis.stats.clone();
    let mut new_landmarks = new_landmark_analysis.landmarks;

    let mut optimized_seed_points = best_two_view
        .as_ref()
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
    if let Some((seed_pair_index, estimate)) = best_two_view.as_ref() {
        let adjustment = multi_view::bundle_adjust(
            &multi_view_analysis,
            &optimized_seed_points,
            &new_landmarks,
            &registered_geometry,
            &features,
            width,
            height,
            focal as f64,
        );
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
            *seed_pair_index,
            estimate,
            &registered_geometry,
            &revisit_context,
        );

        if !closures.is_empty() {
            for closure in &closures {
                if let Some(camera) = registered_geometry
                    .iter_mut()
                    .find(|camera| camera.frame_index == closure.frame_index)
                {
                    camera.rotation = closure.rotation;
                    camera.translation = closure.translation;
                }
            }

            let loop_adjustment = multi_view::bundle_adjust(
                &multi_view_analysis,
                &optimized_seed_points,
                &new_landmarks,
                &registered_geometry,
                &features,
                width,
                height,
                focal as f64,
            );
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
            .filter(|camera| {
                camera.frame_index != *seed_pair_index && camera.frame_index != *seed_pair_index + 1
            })
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

    let dense_sparse_points: Vec<nalgebra::Vector3<f64>> = optimized_seed_points
        .iter()
        .chain(&optimized_new_landmark_positions)
        .copied()
        .collect();
    let dense_analysis = dense::estimate_depth_points(
        &request.frames,
        &registered_geometry,
        &dense_sparse_points,
        focal as f64,
    );
    let dense = dense_analysis.stats;
    let dense_points = dense_analysis.points;
    let multi_view = multi_view_analysis.stats;

    let calibrated_pair = best_two_view.map(|(pair_index, estimate)| {
        let source_features = &features[pair_index];
        let frame = &request.frames[pair_index];
        cameras = vec![
            CameraPose {
                frame_index: pair_index,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                matched_features: 0,
            },
            CameraPose {
                frame_index: pair_index + 1,
                x: estimate.camera_center.x as f32,
                y: estimate.camera_center.y as f32,
                z: estimate.camera_center.z as f32,
                matched_features: estimate.inliers,
            },
        ];
        cameras.extend(registered_cameras.iter().copied());
        cameras.sort_by_key(|camera| camera.frame_index);
        points = estimate
            .points
            .iter()
            .enumerate()
            .map(|(point_index, triangulated)| {
                let feature = &source_features[triangulated.source_feature_index];
                let (r, g, b) = sample_rgb(frame, feature.x, feature.y);
                let descriptor_confidence = (1.0
                    - triangulated.descriptor_distance / options.max_descriptor_distance)
                    .clamp(0.0, 1.0);
                let reprojection_confidence = (1.0
                    / (1.0 + triangulated.reprojection_error_pixels as f32 * 0.5))
                    .clamp(0.15, 1.0);
                let angle_confidence =
                    (triangulated.triangulation_angle_degrees as f32 / 3.0).clamp(0.15, 1.0);
                let position = optimized_seed_points
                    .get(point_index)
                    .copied()
                    .unwrap_or(triangulated.position);
                Point3 {
                    x: position.x as f32,
                    y: position.y as f32,
                    z: position.z as f32,
                    confidence: descriptor_confidence * reprojection_confidence * angle_confidence,
                    r,
                    g,
                    b,
                }
            })
            .collect();
        points.extend(
            new_landmarks
                .iter()
                .enumerate()
                .filter_map(|(new_index, landmark)| {
                    let feature = features
                        .get(landmark.source_frame_index)?
                        .get(landmark.source_feature_index)?;
                    let frame = request.frames.get(landmark.source_frame_index)?;
                    let (r, g, b) = sample_rgb(frame, feature.x, feature.y);
                    let support_confidence =
                        (landmark.supporting_observations as f32 / 4.0).clamp(0.5, 1.0);
                    let reprojection_confidence = (1.0
                        / (1.0 + landmark.median_reprojection_error_pixels as f32 * 0.5))
                        .clamp(0.15, 1.0);
                    let angle_confidence =
                        (landmark.triangulation_angle_degrees as f32 / 3.0).clamp(0.15, 1.0);
                    let position = optimized_new_landmark_positions
                        .get(new_index)
                        .copied()
                        .unwrap_or(landmark.position);
                    Some(Point3 {
                        x: position.x as f32,
                        y: position.y as f32,
                        z: position.z as f32,
                        confidence: support_confidence * reprojection_confidence * angle_confidence,
                        r,
                        g,
                        b,
                    })
                }),
        );

        CalibratedPairStats {
            from_frame: pair_index,
            to_frame: pair_index + 1,
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
    });

    let low_pairs = pairs.iter().filter(|pair| pair.low_parallax).count();
    if low_pairs > pairs.len() / 2 {
        warnings.push(
            "Most frame pairs have weak residual parallax after compensating dominant translation and rotation. Move through a textured scene instead of only panning or rotating the camera."
                .into(),
        );
    }
    if points.len() < 80 {
        warnings.push(
            "The sparse cloud is small. Try a more textured, well-lit scene with slower camera motion."
                .into(),
        );
    }

    if let Some(dense_warning) = dense_warning(&dense) {
        warnings.push(dense_warning);
    }

    let recovered_from_revisit = revisits
        .recoveries
        .iter()
        .filter(|recovery| recovery.accepted)
        .count();
    let closed_drift = revisits
        .closures
        .iter()
        .filter(|closure| closure.accepted)
        .count();
    if !revisits.candidates.is_empty() {
        if recovered_from_revisit > 0 || closed_drift > 0 {
            warnings.push(format!(
                "Bounded non-adjacent revisit screening found {} strong keyframe-pair candidates, recovered {recovered_from_revisit} previously unregistered selected keyframe poses, and retained {closed_drift} seed-anchored registered-pose drift corrections. Revisit evidence changes geometry only through strict seed-landmark PnP and the existing rollback-safe bundle-adjustment acceptance boundary; broader arbitrary non-seed pose-graph closure is still out of scope.",
                revisits.candidates.len()
            ));
        } else if !revisits.recoveries.is_empty() || !revisits.closures.is_empty() {
            warnings.push(format!(
                "Bounded non-adjacent revisit screening found {} strong keyframe-pair candidates and produced recovery or registered-pose closure attempts, but no additional revisit-driven geometry survived the deterministic PnP, bounded-drift, and bundle-adjustment no-regression gates. No camera pose was invented or half-integrated from revisit evidence.",
                revisits.candidates.len()
            ));
        } else {
            warnings.push(format!(
                "Bounded non-adjacent revisit screening found {} strong keyframe-pair candidates, but none had enough accepted seed geometry to recover or close a selected camera pose. No revisit correction was applied.",
                revisits.candidates.len()
            ));
        }
    }

    if calibrated_pair.is_some() {
        let pnp_ready = multi_view
            .registration_candidates
            .iter()
            .filter(|candidate| candidate.pnp_ready)
            .count();
        if registered_views.is_empty() {
            warnings.push(format!(
                "Slice 3 finds {pnp_ready} other selected keyframes with enough tracked seed landmarks for a robust PnP attempt, but neither the initial track-based registration nor bounded direct-revisit recovery produced an additional camera pose that passed the deterministic inlier and reprojection gates. The displayed geometry remains the strongest calibrated adjacent pair; translation scale is arbitrary, and bundle adjustment requires at least one accepted additional view."
            ));
        } else if multi_view.bundle_adjustment.accepted {
            let initial_rmse = multi_view
                .bundle_adjustment
                .initial_rmse_reprojection_error_pixels
                .unwrap_or_default();
            let final_rmse = multi_view
                .bundle_adjustment
                .final_rmse_reprojection_error_pixels
                .unwrap_or_default();
            warnings.push(format!(
                "Slice 3 registered {} additional selected keyframes, including {recovered_from_revisit} recovered from bounded direct revisit evidence, triangulated {} new landmarks, retained {closed_drift} seed-anchored loop corrections, and accepted deterministic bundle adjustment across {} cameras, {} landmarks, and {} supported observations. Reprojection RMSE improved from {:.2} px to {:.2} px while the calibrated seed-pair cameras remained fixed to preserve the arbitrary monocular gauge. No metric-scale claim or arbitrary non-seed pose-graph optimization is made.",
                registered_views.len(),
                multi_view.new_landmarks.accepted_landmarks,
                multi_view.bundle_adjustment.optimized_cameras,
                multi_view.bundle_adjustment.optimized_landmarks,
                multi_view.bundle_adjustment.observations,
                initial_rmse,
                final_rmse
            ));
        } else if multi_view.bundle_adjustment.attempted {
            warnings.push(format!(
                "Slice 3 registered {} additional selected keyframes, including {recovered_from_revisit} recovered from bounded direct revisit evidence, and triangulated {} new landmarks. Bundle adjustment ran across {} supported observations but its candidate geometry did not satisfy the no-regression acceptance boundary, so the pre-adjustment cameras and landmarks were retained and no loop correction was committed. The calibrated seed pair remains the fixed arbitrary monocular gauge.",
                registered_views.len(),
                multi_view.new_landmarks.accepted_landmarks,
                multi_view.bundle_adjustment.observations
            ));
        } else {
            warnings.push(format!(
                "Slice 3 registered {} additional selected keyframes, including {recovered_from_revisit} recovered from bounded direct revisit evidence, and triangulated {} new landmarks, but there were not enough mutually supported registered observations to run bundle adjustment. No loop correction can be retained without that downstream gate; all geometry remains in the seed pair's arbitrary monocular coordinate frame.",
                registered_views.len(),
                multi_view.new_landmarks.accepted_landmarks
            ));
        }
    } else {
        warnings.push(
            "No adjacent pair passed the calibrated epipolar, cheirality, reprojection, and triangulation-angle gates, so this result retains the conservative uncalibrated MVP preview."
                .into(),
        );
    }

    Ok(ReconstructionResult {
        cameras,
        points,
        dense_points,
        dense,
        pairs,
        calibrated_pair,
        multi_view,
        revisits,
        registered_views,
        warnings,
    })
}

fn dense_warning(dense: &DenseStats) -> Option<String> {
    if !dense.attempted {
        return None;
    }

    let reference_frame = dense.reference_frame.map_or(0, |frame| frame + 1);
    if dense.accepted_points > 0 {
        return Some(format!(
            "Slice 4 dense point fusion accepted {} fused scene points from reference frame {} using {} registered source views and {} inverse-depth hypotheses. {} primary candidates passed reciprocal depth consistency; spatial fusion rejected {} reverse observations. Fused points remain separate from the sparse map; meshing, general multi-reference depth aggregation, and metric scale are not claimed yet.",
            dense.accepted_points,
            reference_frame,
            dense.source_views,
            dense.depth_hypotheses,
            dense.reciprocal_consistent_points,
            dense.fusion_rejected_observations
        ));
    }

    if dense.reciprocal_consistent_points > 0 {
        return Some(format!(
            "Slice 4 dense point fusion ran from reference frame {} with {} registered source views. {} primary candidates passed reciprocal depth consistency, but no fused point survived the spatial-consistency and minimum-observation gates; {} reverse observations were rejected as spatially inconsistent. No dense geometry was invented.",
            reference_frame,
            dense.source_views,
            dense.reciprocal_consistent_points,
            dense.fusion_rejected_observations
        ));
    }

    if dense.reciprocal_checked_points > 0 {
        return Some(format!(
            "Slice 4 coarse depth estimation ran from reference frame {} with {} registered source views. {} primary candidates passed the texture, photometric-error, and ambiguity gates, but reciprocal depth consistency rejected all of them. No dense geometry was invented.",
            reference_frame,
            dense.source_views,
            dense.reciprocal_checked_points
        ));
    }

    Some(format!(
        "Slice 4 coarse depth estimation ran from reference frame {} with {} registered source views, but no sampled pixel passed the texture, photometric-error, and ambiguity gates. No dense geometry was invented.",
        reference_frame, dense.source_views
    ))
}

fn validate_request(request: &ReconstructionRequest) -> Result<(), String> {
    if request.frames.len() < 2 {
        return Err("at least two sampled frames are required".into());
    }
    if request.options.max_descriptor_distance <= 0.0 {
        return Err("max descriptor distance must be positive".into());
    }
    if request
        .options
        .focal_length_pixels
        .is_some_and(|focal| !focal.is_finite() || focal <= 0.0)
    {
        return Err("focal length in pixels must be finite and positive".into());
    }

    let width = request.frames[0].width;
    let height = request.frames[0].height;
    if width < 32 || height < 24 {
        return Err("sampled frames are too small for reconstruction".into());
    }

    for frame in &request.frames {
        if frame.width != width || frame.height != height {
            return Err("all sampled frames must share dimensions".into());
        }
        let expected = width as usize * height as usize * 4;
        if frame.rgba.len() != expected {
            return Err(format!(
                "invalid RGBA payload: expected {expected} bytes, got {}",
                frame.rgba.len()
            ));
        }
    }
    Ok(())
}

fn compensate_global_motion(
    a: &[Feature],
    b: &[Feature],
    matches: &[FeatureMatch],
    width: u32,
    height: u32,
) -> (f32, f32, Vec<f32>) {
    if matches.is_empty() {
        return (0.0, 0.0, Vec::new());
    }

    let center_x = width as f32 * 0.5;
    let center_y = height as f32 * 0.5;
    let mut raw_dx = Vec::with_capacity(matches.len());
    let mut raw_dy = Vec::with_capacity(matches.len());
    for feature_match in matches {
        let source = &a[feature_match.a];
        let target = &b[feature_match.b];
        raw_dx.push(target.x as f32 - source.x as f32);
        raw_dy.push(target.y as f32 - source.y as f32);
    }

    let mut initial_dx = raw_dx.clone();
    let mut initial_dy = raw_dy.clone();
    let initial_tx = median(&mut initial_dx);
    let initial_ty = median(&mut initial_dy);
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (index, feature_match) in matches.iter().enumerate() {
        let source = &a[feature_match.a];
        let x = source.x as f32 - center_x;
        let y = source.y as f32 - center_y;
        let dx = raw_dx[index] - initial_tx;
        let dy = raw_dy[index] - initial_ty;
        numerator += x * dy - y * dx;
        denominator += x * x + y * y;
    }
    let rotation = if denominator > f32::EPSILON {
        (numerator / denominator).clamp(-0.35, 0.35)
    } else {
        0.0
    };

    let mut corrected_dx = Vec::with_capacity(matches.len());
    let mut corrected_dy = Vec::with_capacity(matches.len());
    for (index, feature_match) in matches.iter().enumerate() {
        let source = &a[feature_match.a];
        let x = source.x as f32 - center_x;
        let y = source.y as f32 - center_y;
        corrected_dx.push(raw_dx[index] + rotation * y);
        corrected_dy.push(raw_dy[index] - rotation * x);
    }
    let translation_x = median(&mut corrected_dx);
    let translation_y = median(&mut corrected_dy);

    let residuals = matches
        .iter()
        .enumerate()
        .map(|(index, feature_match)| {
            let source = &a[feature_match.a];
            let x = source.x as f32 - center_x;
            let y = source.y as f32 - center_y;
            let predicted_dx = translation_x - rotation * y;
            let predicted_dy = translation_y + rotation * x;
            (raw_dx[index] - predicted_dx).hypot(raw_dy[index] - predicted_dy)
        })
        .collect();

    (translation_x, translation_y, residuals)
}

fn to_luma(frame: &FrameInput) -> Vec<u8> {
    frame
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            ((77 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16) >> 8) as u8
        })
        .collect()
}

fn detect_features(
    luma: &[u8],
    width: u32,
    height: u32,
    options: ReconstructionOptions,
) -> Vec<Feature> {
    let border = options.descriptor_radius.max(3) + 2;
    let mut candidates = Vec::new();

    for y in border..height.saturating_sub(border) {
        for x in border..width.saturating_sub(border) {
            let mut sxx = 0.0f32;
            let mut syy = 0.0f32;
            let mut sxy = 0.0f32;
            for wy in y - 1..=y + 1 {
                for wx in x - 1..=x + 1 {
                    let gx = sample_luma(luma, width, wx + 1, wy) as f32
                        - sample_luma(luma, width, wx - 1, wy) as f32;
                    let gy = sample_luma(luma, width, wx, wy + 1) as f32
                        - sample_luma(luma, width, wx, wy - 1) as f32;
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            let score = det - 0.04 * trace * trace;
            if score > 1_000_000.0 {
                candidates.push((x, y, score));
            }
        }
    }

    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
    let min_distance_sq = (options.min_feature_distance * options.min_feature_distance) as i64;
    let mut selected: Vec<Feature> = Vec::with_capacity(options.max_features);

    for (x, y, score) in candidates {
        if selected.iter().any(|feature| {
            let dx = feature.x as i64 - x as i64;
            let dy = feature.y as i64 - y as i64;
            dx * dx + dy * dy < min_distance_sq
        }) {
            continue;
        }

        selected.push(Feature {
            x,
            y,
            score,
            descriptor: descriptor(luma, width, x, y, options.descriptor_radius),
        });
        if selected.len() >= options.max_features {
            break;
        }
    }

    selected
}

fn descriptor(luma: &[u8], width: u32, x: u32, y: u32, radius: u32) -> Vec<i16> {
    let side = radius * 2 + 1;
    let mut values = Vec::with_capacity((side * side) as usize);
    let mut sum = 0i32;
    for py in y - radius..=y + radius {
        for px in x - radius..=x + radius {
            let value = sample_luma(luma, width, px, py) as i16;
            values.push(value);
            sum += value as i32;
        }
    }
    let mean = sum / values.len() as i32;
    for value in &mut values {
        *value -= mean as i16;
    }
    values
}

fn match_features(
    a: &[Feature],
    b: &[Feature],
    options: ReconstructionOptions,
) -> Vec<FeatureMatch> {
    let radius_sq = (options.match_radius * options.match_radius) as i64;
    let mut proposals = Vec::new();

    for (a_index, feature_a) in a.iter().enumerate() {
        let mut best: Option<(usize, f32)> = None;
        let mut second = f32::INFINITY;

        for (b_index, feature_b) in b.iter().enumerate() {
            let dx = feature_b.x as i64 - feature_a.x as i64;
            let dy = feature_b.y as i64 - feature_a.y as i64;
            if dx * dx + dy * dy > radius_sq {
                continue;
            }
            let distance = descriptor_distance(&feature_a.descriptor, &feature_b.descriptor);
            match best {
                None => best = Some((b_index, distance)),
                Some((_, best_distance)) if distance < best_distance => {
                    second = best_distance;
                    best = Some((b_index, distance));
                }
                Some(_) if distance < second => second = distance,
                _ => {}
            }
        }

        if let Some((b_index, best_distance)) = best {
            let ratio_ok = second.is_infinite() || best_distance < second * options.ratio_threshold;
            if ratio_ok && best_distance <= options.max_descriptor_distance {
                proposals.push(FeatureMatch {
                    a: a_index,
                    b: b_index,
                    distance: best_distance,
                });
            }
        }
    }

    proposals.sort_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b[right.b]
                    .score
                    .partial_cmp(&b[left.b].score)
                    .unwrap_or(Ordering::Equal)
            })
    });
    let mut used_b = HashSet::new();
    proposals
        .into_iter()
        .filter(|proposal| used_b.insert(proposal.b))
        .collect()
}

fn descriptor_distance(a: &[i16], b: &[i16]) -> f32 {
    let total: i32 = a
        .iter()
        .zip(b)
        .map(|(left, right)| (*left as i32 - *right as i32).abs())
        .sum();
    total as f32 / a.len() as f32
}

fn sample_luma(luma: &[u8], width: u32, x: u32, y: u32) -> u8 {
    luma[y as usize * width as usize + x as usize]
}

fn sample_rgb(frame: &FrameInput, x: u32, y: u32) -> (u8, u8, u8) {
    let index = (y as usize * frame.width as usize + x as usize) * 4;
    (
        frame.rgba[index],
        frame.rgba[index + 1],
        frame.rgba[index + 2],
    )
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
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

    fn synthetic_frame(width: u32, height: u32, shift_x: i32) -> FrameInput {
        let mut rgba = vec![18u8; width as usize * height as usize * 4];
        for pixel in rgba.as_chunks_mut::<4>().0 {
            pixel[3] = 255;
        }

        for (index, (base_x, base_y)) in [
            (18, 16),
            (37, 21),
            (58, 15),
            (24, 39),
            (48, 44),
            (70, 36),
            (15, 58),
            (41, 62),
            (66, 57),
        ]
        .into_iter()
        .enumerate()
        {
            let depth_shift = (index as i32 % 3) * shift_x / 3;
            let x = base_x + shift_x + depth_shift;
            if x < 3 || x >= width as i32 - 4 || base_y < 3 || base_y >= height as i32 - 4 {
                continue;
            }
            let intensity = 90 + index as u8 * 16;
            for py in base_y - 2..=base_y + 2 {
                for px in x - 2..=x + 2 {
                    let edge = px == x - 2 || px == x + 2 || py == base_y - 2 || py == base_y + 2;
                    let value = if edge { 245 } else { intensity };
                    let offset = (py as usize * width as usize + px as usize) * 4;
                    rgba[offset] = value;
                    rgba[offset + 1] = value.saturating_sub(index as u8 * 2);
                    rgba[offset + 2] = value / 2;
                }
            }
        }

        FrameInput {
            width,
            height,
            rgba,
        }
    }

    #[test]
    fn detects_repeatable_corners() {
        let frame = synthetic_frame(96, 80, 0);
        let luma = to_luma(&frame);
        let features = detect_features(
            &luma,
            frame.width,
            frame.height,
            ReconstructionOptions::default(),
        );
        assert!(
            features.len() >= 12,
            "found only {} features",
            features.len()
        );
    }

    #[test]
    fn reconstructs_parallax_sequence_into_sparse_output() {
        let request = ReconstructionRequest {
            frames: vec![
                synthetic_frame(96, 80, 0),
                synthetic_frame(96, 80, 3),
                synthetic_frame(96, 80, 6),
            ],
            options: ReconstructionOptions {
                match_radius: 12,
                max_descriptor_distance: 55.0,
                ..ReconstructionOptions::default()
            },
        };

        let result = reconstruct(&request).expect("reconstruction should succeed");
        assert!(result.cameras.len() >= 2);
        assert_eq!(result.pairs.len(), 2);
        assert!(
            result.pairs[0].matches >= 6,
            "too few matches: {}",
            result.pairs[0].matches
        );
        assert!(result.pairs[0].overlap_ratio > 0.0);
        assert!(result.pairs[0].median_dx > 1.0);
        assert!(result.pairs[0].median_parallax_residual > 0.5);
        assert!(!result.pairs[0].low_parallax);
        assert!(result.multi_view.track_count > 0);
        assert!(!result.points.is_empty());
    }

    #[test]
    fn stationary_sequence_does_not_invent_camera_motion() {
        let frame = synthetic_frame(96, 80, 0);
        let request = ReconstructionRequest {
            frames: vec![frame.clone(), frame],
            options: ReconstructionOptions {
                max_descriptor_distance: 55.0,
                ..ReconstructionOptions::default()
            },
        };

        let result = reconstruct(&request).expect("stationary reconstruction should succeed");
        assert!(result.pairs[0].low_parallax);
        assert!(result.calibrated_pair.is_none());
        assert_eq!(result.multi_view.keyframes, vec![0]);
        assert!(result.multi_view.registration_candidates.is_empty());
        assert!(result.revisits.candidates.is_empty());
        assert!(result.revisits.recoveries.is_empty());
        assert!(result.revisits.closures.is_empty());
        assert!(result.registered_views.is_empty());
        assert!(!result.multi_view.bundle_adjustment.attempted);
        assert!(!result.dense.attempted);
        assert!(result.dense_points.is_empty());
        assert!(result.cameras[1].x.abs() < f32::EPSILON);
        assert!(result.cameras[1].y.abs() < f32::EPSILON);
        assert!(result.cameras[1].z.abs() < f32::EPSILON);
        assert!(result.points.is_empty());
    }

    #[test]
    fn dominant_rotation_is_not_counted_as_parallax() {
        let center_x = 48.0f32;
        let center_y = 40.0f32;
        let angle = 0.05f32;
        let translation_x = 2.0f32;
        let translation_y = -1.0f32;
        let source_positions = [
            (20u32, 18u32),
            (48, 15),
            (75, 20),
            (24, 40),
            (70, 42),
            (18, 64),
            (48, 66),
            (76, 62),
        ];
        let source: Vec<Feature> = source_positions
            .iter()
            .map(|&(x, y)| Feature {
                x,
                y,
                score: 1.0,
                descriptor: vec![0],
            })
            .collect();
        let target: Vec<Feature> = source_positions
            .iter()
            .map(|&(x, y)| {
                let centered_x = x as f32 - center_x;
                let centered_y = y as f32 - center_y;
                Feature {
                    x: (x as f32 + translation_x - angle * centered_y).round() as u32,
                    y: (y as f32 + translation_y + angle * centered_x).round() as u32,
                    score: 1.0,
                    descriptor: vec![0],
                }
            })
            .collect();
        let matches: Vec<FeatureMatch> = (0..source.len())
            .map(|index| FeatureMatch {
                a: index,
                b: index,
                distance: 0.0,
            })
            .collect();

        let (_, _, mut residuals) = compensate_global_motion(&source, &target, &matches, 96, 80);
        let residual = median(&mut residuals);
        assert!(residual < 0.55, "rotation residual was {residual}");
    }

    #[test]
    fn dense_warning_reports_successful_fusion_without_stale_future_claims() {
        let dense = DenseStats {
            attempted: true,
            reference_frame: Some(1),
            source_views: 3,
            depth_hypotheses: 24,
            accepted_points: 17,
            reciprocal_consistent_points: 19,
            fusion_rejected_observations: 4,
            ..DenseStats::default()
        };

        let warning = dense_warning(&dense).expect("dense warning");
        assert!(warning.contains("17 fused scene points"));
        assert!(warning.contains("19 primary candidates passed reciprocal depth consistency"));
        assert!(warning.contains("spatial fusion rejected 4 reverse observations"));
        assert!(!warning.contains("fusion remain future"));
        assert!(!warning.contains("multi-view depth consistency"));
    }

    #[test]
    fn dense_warning_distinguishes_fusion_rejection_from_earlier_gates() {
        let dense = DenseStats {
            attempted: true,
            reference_frame: Some(0),
            source_views: 2,
            reciprocal_checked_points: 9,
            reciprocal_rejected_points: 3,
            reciprocal_consistent_points: 6,
            fusion_input_observations: 14,
            fusion_rejected_observations: 8,
            fusion_rejected_points: 6,
            ..DenseStats::default()
        };

        let warning = dense_warning(&dense).expect("dense warning");
        assert!(warning.contains("6 primary candidates passed reciprocal depth consistency"));
        assert!(warning.contains("no fused point survived"));
        assert!(warning.contains("8 reverse observations were rejected"));
        assert!(!warning.contains("no sampled pixel passed"));
    }

    #[test]
    fn rejects_invalid_focal_length() {
        let request = ReconstructionRequest {
            frames: vec![synthetic_frame(96, 80, 0), synthetic_frame(96, 80, 3)],
            options: ReconstructionOptions {
                focal_length_pixels: Some(0.0),
                ..ReconstructionOptions::default()
            },
        };
        assert!(reconstruct(&request).is_err());
    }

    #[test]
    fn rejects_malformed_rgba_payloads() {
        let request = ReconstructionRequest {
            frames: vec![
                FrameInput {
                    width: 96,
                    height: 80,
                    rgba: vec![0; 4],
                },
                synthetic_frame(96, 80, 0),
            ],
            options: ReconstructionOptions::default(),
        };

        assert!(reconstruct(&request).is_err());
    }
}
