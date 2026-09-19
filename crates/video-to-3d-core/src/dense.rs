use crate::{multi_view::RegisteredCamera, FrameInput, Point3};
use nalgebra::Vector3;
use serde::Serialize;
use std::collections::BTreeSet;

const MAX_REFERENCE_VIEWS: usize = 3;
const MAX_REFERENCE_ATTEMPTS: usize = 6;
const MIN_REFERENCE_PATCH_POINTS: usize = 3;

mod legacy {
    pub(super) fn minimum_visible_sparse_points() -> usize {
        MIN_VISIBLE_SPARSE_POINTS
    }

    pub(super) fn visible_sparse_points_for_reference(
        camera: &RegisteredCamera,
        sparse_points: &[Vector3<f64>],
        width: u32,
        height: u32,
        focal: f64,
    ) -> Vec<Vector3<f64>> {
        sparse_points
            .iter()
            .copied()
            .filter(|point| {
                project(camera, *point, width, height, focal)
                    .is_some_and(|(x, y, _)| in_image_bounds(x, y, width, height))
            })
            .collect()
    }

    include!("dense_legacy.rs");
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DenseGridSite {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DenseReferenceAttemptStats {
    pub reference_frame: usize,
    pub attempted: bool,
    pub accepted: bool,
    pub skip_reason: Option<String>,
    pub sampled_pixels: usize,
    pub accepted_points: usize,
    pub reciprocal_checked_points: usize,
    pub reciprocal_rejected_points: usize,
    pub reciprocal_consistent_points: usize,
    pub surface_completion_proposals: usize,
    pub surface_completed_points: usize,
    pub surface_completion_rejected_texture: usize,
    pub surface_completion_rejected_cross_view: usize,
    pub surface_completion_rejected_reciprocal: usize,
    pub surface_completion_rejected_fusion: usize,
    pub surface_completion_rejected_footprint: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DenseReferencePatchStats {
    pub reference_frame: usize,
    pub source_frames: Vec<usize>,
    pub primary_start: usize,
    pub primary_points: usize,
    pub completion_start: usize,
    pub completed_points: usize,
    pub sampled_pixels: usize,
    pub accepted_points: usize,
    pub reciprocal_consistent_points: usize,
    pub grid_stride: usize,
    pub grid_border: u32,
    pub search_min_depth: Option<f32>,
    pub search_max_depth: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DenseStats {
    pub attempted: bool,
    pub skip_reason: Option<String>,
    pub reference_frame: Option<usize>,
    pub source_views: usize,
    pub source_frames: Vec<usize>,
    pub sampled_pixels: usize,
    pub depth_hypotheses: usize,
    pub accepted_points: usize,
    pub surface_completion_proposals: usize,
    pub surface_completed_points: usize,
    pub surface_completion_rejected_texture: usize,
    pub surface_completion_rejected_cross_view: usize,
    pub surface_completion_rejected_reciprocal: usize,
    pub surface_completion_rejected_fusion: usize,
    pub surface_completion_rejected_footprint: usize,
    pub grid_stride: usize,
    pub grid_border: u32,
    pub reciprocal_checked_points: usize,
    pub reciprocal_rejected_points: usize,
    pub reciprocal_consistent_points: usize,
    pub fusion_input_observations: usize,
    pub fusion_rejected_observations: usize,
    pub fusion_rejected_points: usize,
    pub median_fusion_observations: Option<f32>,
    pub median_supporting_views: Option<f32>,
    pub median_photometric_error: Option<f32>,
    pub search_min_depth: Option<f32>,
    pub search_max_depth: Option<f32>,
    pub reference_frames: Vec<usize>,
    pub reference_attempts: Vec<DenseReferenceAttemptStats>,
    pub reference_patches: Vec<DenseReferencePatchStats>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct DenseAnalysis {
    pub stats: DenseStats,
    pub points: Vec<Point3>,
    pub grid_sites: Vec<DenseGridSite>,
}

struct AcceptedPatch {
    reference_frame: usize,
    stats: legacy::DenseStats,
    primary_points: Vec<Point3>,
    primary_sites: Vec<DenseGridSite>,
    completion_points: Vec<Point3>,
    completion_sites: Vec<DenseGridSite>,
}

pub(super) fn estimate_depth_points(
    frames: &[FrameInput],
    cameras: &[RegisteredCamera],
    sparse_points: &[Vector3<f64>],
    focal: f64,
) -> DenseAnalysis {
    let reference_order = reference_candidate_order(frames, cameras, sparse_points, focal);
    if reference_order.is_empty() {
        return from_legacy(legacy::estimate_depth_points(
            frames,
            cameras,
            sparse_points,
            focal,
        ));
    }

    let mut patches = Vec::new();
    let mut attempts = Vec::new();
    for camera_index in reference_order.into_iter().take(MAX_REFERENCE_ATTEMPTS) {
        let Some(analysis) =
            reconstruct_reference_patch(frames, cameras, sparse_points, focal, camera_index)
        else {
            continue;
        };

        let reference_frame = cameras[camera_index].frame_index;
        let accepted = analysis.stats.attempted
            && analysis.points.len() >= MIN_REFERENCE_PATCH_POINTS;
        attempts.push(reference_attempt_stats(
            reference_frame,
            &analysis.stats,
            analysis.points.len(),
            accepted,
        ));
        if !accepted {
            continue;
        }

        patches.push(split_patch(reference_frame, analysis));
        if patches.len() >= MAX_REFERENCE_VIEWS {
            break;
        }
    }

    if patches.is_empty() {
        let mut fallback = from_legacy(legacy::estimate_depth_points(
            frames,
            cameras,
            sparse_points,
            focal,
        ));
        let fallback_attempts = std::mem::take(&mut fallback.stats.reference_attempts);
        fallback.stats.reference_attempts = merge_fallback_attempts(attempts, fallback_attempts);
        return fallback;
    }

    combine_patches(patches, attempts)
}

fn reference_candidate_order(
    frames: &[FrameInput],
    cameras: &[RegisteredCamera],
    sparse_points: &[Vector3<f64>],
    focal: f64,
) -> Vec<usize> {
    let minimum_visible = legacy::minimum_visible_sparse_points();
    let mut candidates = cameras
        .iter()
        .enumerate()
        .filter_map(|(camera_index, camera)| {
            let frame = frames.get(camera.frame_index)?;
            let visible = legacy::visible_sparse_points_for_reference(
                camera,
                sparse_points,
                frame.width,
                frame.height,
                focal,
            )
            .len();
            (visible >= minimum_visible).then_some((camera_index, visible))
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|(left_index, left_visible), (right_index, right_visible)| {
        right_visible.cmp(left_visible).then_with(|| {
            cameras[*left_index]
                .frame_index
                .cmp(&cameras[*right_index].frame_index)
        })
    });
    let Some(&(primary_index, _)) = candidates.first() else {
        return Vec::new();
    };
    let primary_center = cameras[primary_index].camera_center();
    candidates[1..].sort_by(|(left_index, left_visible), (right_index, right_visible)| {
        let left_distance = (cameras[*left_index].camera_center() - primary_center).norm_squared();
        let right_distance =
            (cameras[*right_index].camera_center() - primary_center).norm_squared();
        right_distance
            .total_cmp(&left_distance)
            .then_with(|| right_visible.cmp(left_visible))
            .then_with(|| {
                cameras[*left_index]
                    .frame_index
                    .cmp(&cameras[*right_index].frame_index)
            })
    });
    candidates
        .into_iter()
        .map(|(camera_index, _)| camera_index)
        .collect()
}

fn reconstruct_reference_patch(
    frames: &[FrameInput],
    cameras: &[RegisteredCamera],
    sparse_points: &[Vector3<f64>],
    focal: f64,
    reference_camera_index: usize,
) -> Option<legacy::DenseAnalysis> {
    let reference_camera = cameras.get(reference_camera_index)?;
    let reference_frame = frames.get(reference_camera.frame_index)?;
    let visible_sparse = legacy::visible_sparse_points_for_reference(
        reference_camera,
        sparse_points,
        reference_frame.width,
        reference_frame.height,
        focal,
    );
    if visible_sparse.len() < legacy::minimum_visible_sparse_points() {
        return None;
    }

    let mut source_indices = (0..cameras.len())
        .filter(|camera_index| *camera_index != reference_camera_index)
        .filter(|camera_index| frames.get(cameras[*camera_index].frame_index).is_some())
        .collect::<Vec<_>>();
    source_indices.sort_by_key(|camera_index| {
        (
            cameras[*camera_index]
                .frame_index
                .abs_diff(reference_camera.frame_index),
            cameras[*camera_index].frame_index,
        )
    });

    let mut camera_order = Vec::with_capacity(1 + source_indices.len());
    camera_order.push(reference_camera_index);
    camera_order.extend(source_indices);

    let mut remapped_frames = Vec::with_capacity(camera_order.len());
    let mut remapped_cameras = Vec::with_capacity(camera_order.len());
    for (remapped_index, camera_index) in camera_order.iter().copied().enumerate() {
        let camera = cameras.get(camera_index)?;
        remapped_frames.push(frames.get(camera.frame_index)?.clone());
        let mut remapped_camera = camera.clone();
        remapped_camera.frame_index = remapped_index;
        remapped_cameras.push(remapped_camera);
    }

    let mut analysis = legacy::estimate_depth_points(
        &remapped_frames,
        &remapped_cameras,
        &visible_sparse,
        focal,
    );
    analysis.stats.reference_frame = Some(reference_camera.frame_index);
    analysis.stats.source_frames = analysis
        .stats
        .source_frames
        .iter()
        .filter_map(|remapped_index| {
            camera_order
                .get(*remapped_index)
                .and_then(|camera_index| cameras.get(*camera_index))
                .map(|camera| camera.frame_index)
        })
        .collect();
    Some(analysis)
}

fn reference_attempt_stats(
    reference_frame: usize,
    stats: &legacy::DenseStats,
    accepted_points: usize,
    accepted: bool,
) -> DenseReferenceAttemptStats {
    let skip_reason = if accepted {
        None
    } else if let Some(reason) = &stats.skip_reason {
        Some(reason.clone())
    } else if !stats.attempted {
        Some("dense estimation did not run for this reference".into())
    } else {
        Some(format!(
            "reference produced {accepted_points} accepted dense points; at least {MIN_REFERENCE_PATCH_POINTS} are required for a surface patch"
        ))
    };

    DenseReferenceAttemptStats {
        reference_frame,
        attempted: stats.attempted,
        accepted,
        skip_reason,
        sampled_pixels: stats.sampled_pixels,
        accepted_points,
        reciprocal_checked_points: stats.reciprocal_checked_points,
        reciprocal_rejected_points: stats.reciprocal_rejected_points,
        reciprocal_consistent_points: stats.reciprocal_consistent_points,
        surface_completion_proposals: stats.surface_completion_proposals,
        surface_completed_points: stats.surface_completed_points,
        surface_completion_rejected_texture: stats.surface_completion_rejected_texture,
        surface_completion_rejected_cross_view: stats.surface_completion_rejected_cross_view,
        surface_completion_rejected_reciprocal: stats.surface_completion_rejected_reciprocal,
        surface_completion_rejected_fusion: stats.surface_completion_rejected_fusion,
        surface_completion_rejected_footprint: stats.surface_completion_rejected_footprint,
    }
}

fn merge_fallback_attempts(
    mut attempts: Vec<DenseReferenceAttemptStats>,
    fallback_attempts: Vec<DenseReferenceAttemptStats>,
) -> Vec<DenseReferenceAttemptStats> {
    for fallback_attempt in fallback_attempts {
        if let Some(existing) = attempts
            .iter_mut()
            .find(|attempt| attempt.reference_frame == fallback_attempt.reference_frame)
        {
            if fallback_attempt.accepted && !existing.accepted {
                *existing = fallback_attempt;
            }
        } else {
            attempts.push(fallback_attempt);
        }
    }
    attempts
}

fn convert_grid_sites(sites: Vec<legacy::DenseGridSite>) -> Vec<DenseGridSite> {
    sites
        .into_iter()
        .map(|site| DenseGridSite {
            x: site.x,
            y: site.y,
        })
        .collect()
}

fn split_patch(reference_frame: usize, analysis: legacy::DenseAnalysis) -> AcceptedPatch {
    let legacy::DenseAnalysis {
        stats,
        mut points,
        grid_sites,
    } = analysis;
    let mut grid_sites = convert_grid_sites(grid_sites);
    let completion_count = stats.surface_completed_points.min(points.len());
    let primary_count = points.len() - completion_count;
    let completion_points = points.split_off(primary_count);
    let completion_sites = grid_sites.split_off(primary_count);

    AcceptedPatch {
        reference_frame,
        stats,
        primary_points: points,
        primary_sites: grid_sites,
        completion_points,
        completion_sites,
    }
}

fn combine_patches(
    patches: Vec<AcceptedPatch>,
    reference_attempts: Vec<DenseReferenceAttemptStats>,
) -> DenseAnalysis {
    let total_primary = patches
        .iter()
        .map(|patch| patch.primary_points.len())
        .sum::<usize>();
    let total_completion = patches
        .iter()
        .map(|patch| patch.completion_points.len())
        .sum::<usize>();
    let mut points = Vec::with_capacity(total_primary + total_completion);
    let mut grid_sites = Vec::with_capacity(total_primary + total_completion);
    let mut primary_starts = Vec::with_capacity(patches.len());
    for patch in &patches {
        primary_starts.push(points.len());
        points.extend_from_slice(&patch.primary_points);
        grid_sites.extend_from_slice(&patch.primary_sites);
    }
    let mut completion_starts = Vec::with_capacity(patches.len());
    for patch in &patches {
        completion_starts.push(points.len());
        points.extend_from_slice(&patch.completion_points);
        grid_sites.extend_from_slice(&patch.completion_sites);
    }

    let reference_frames = patches
        .iter()
        .map(|patch| patch.reference_frame)
        .collect::<Vec<_>>();
    let primary_reference = reference_frames[0];
    let mut dense_participating_frames = BTreeSet::new();
    for patch in &patches {
        dense_participating_frames.insert(patch.reference_frame);
        dense_participating_frames.extend(patch.stats.source_frames.iter().copied());
    }
    dense_participating_frames.remove(&primary_reference);
    let source_frames = dense_participating_frames.into_iter().collect::<Vec<_>>();

    let reference_patches = patches
        .iter()
        .enumerate()
        .map(|(index, patch)| DenseReferencePatchStats {
            reference_frame: patch.reference_frame,
            source_frames: patch.stats.source_frames.clone(),
            primary_start: primary_starts[index],
            primary_points: patch.primary_points.len(),
            completion_start: completion_starts[index],
            completed_points: patch.completion_points.len(),
            sampled_pixels: patch.stats.sampled_pixels,
            accepted_points: patch.primary_points.len() + patch.completion_points.len(),
            reciprocal_consistent_points: patch.stats.reciprocal_consistent_points,
            grid_stride: patch.stats.grid_stride,
            grid_border: patch.stats.grid_border,
            search_min_depth: patch.stats.search_min_depth,
            search_max_depth: patch.stats.search_max_depth,
        })
        .collect::<Vec<_>>();

    let first = &patches[0].stats;
    let single_reference = patches.len() == 1;
    DenseAnalysis {
        stats: DenseStats {
            attempted: true,
            skip_reason: None,
            reference_frame: Some(primary_reference),
            source_views: source_frames.len(),
            source_frames,
            sampled_pixels: patches.iter().map(|patch| patch.stats.sampled_pixels).sum(),
            depth_hypotheses: patches
                .iter()
                .map(|patch| patch.stats.depth_hypotheses)
                .max()
                .unwrap_or_default(),
            accepted_points: points.len(),
            surface_completion_proposals: patches
                .iter()
                .map(|patch| patch.stats.surface_completion_proposals)
                .sum(),
            surface_completed_points: total_completion,
            surface_completion_rejected_texture: patches
                .iter()
                .map(|patch| patch.stats.surface_completion_rejected_texture)
                .sum(),
            surface_completion_rejected_cross_view: patches
                .iter()
                .map(|patch| patch.stats.surface_completion_rejected_cross_view)
                .sum(),
            surface_completion_rejected_reciprocal: patches
                .iter()
                .map(|patch| patch.stats.surface_completion_rejected_reciprocal)
                .sum(),
            surface_completion_rejected_fusion: patches
                .iter()
                .map(|patch| patch.stats.surface_completion_rejected_fusion)
                .sum(),
            surface_completion_rejected_footprint: patches
                .iter()
                .map(|patch| patch.stats.surface_completion_rejected_footprint)
                .sum(),
            grid_stride: first.grid_stride,
            grid_border: first.grid_border,
            reciprocal_checked_points: patches
                .iter()
                .map(|patch| patch.stats.reciprocal_checked_points)
                .sum(),
            reciprocal_rejected_points: patches
                .iter()
                .map(|patch| patch.stats.reciprocal_rejected_points)
                .sum(),
            reciprocal_consistent_points: patches
                .iter()
                .map(|patch| patch.stats.reciprocal_consistent_points)
                .sum(),
            fusion_input_observations: patches
                .iter()
                .map(|patch| patch.stats.fusion_input_observations)
                .sum(),
            fusion_rejected_observations: patches
                .iter()
                .map(|patch| patch.stats.fusion_rejected_observations)
                .sum(),
            fusion_rejected_points: patches
                .iter()
                .map(|patch| patch.stats.fusion_rejected_points)
                .sum(),
            median_fusion_observations: single_reference
                .then_some(first.median_fusion_observations)
                .flatten(),
            median_supporting_views: single_reference
                .then_some(first.median_supporting_views)
                .flatten(),
            median_photometric_error: single_reference
                .then_some(first.median_photometric_error)
                .flatten(),
            search_min_depth: single_reference.then_some(first.search_min_depth).flatten(),
            search_max_depth: single_reference.then_some(first.search_max_depth).flatten(),
            reference_frames,
            reference_attempts,
            reference_patches,
        },
        points,
        grid_sites,
    }
}

fn from_legacy(analysis: legacy::DenseAnalysis) -> DenseAnalysis {
    let legacy::DenseAnalysis {
        stats,
        points,
        grid_sites,
    } = analysis;
    let completion_count = stats.surface_completed_points.min(points.len());
    let primary_count = points.len() - completion_count;
    let reference_frames = stats.reference_frame.into_iter().collect::<Vec<_>>();
    let reference_attempts = stats
        .reference_frame
        .map(|reference_frame| {
            vec![reference_attempt_stats(
                reference_frame,
                &stats,
                points.len(),
                stats.attempted && points.len() >= MIN_REFERENCE_PATCH_POINTS,
            )]
        })
        .unwrap_or_default();
    let reference_patches = stats
        .reference_frame
        .filter(|_| stats.attempted)
        .map(|reference_frame| {
            vec![DenseReferencePatchStats {
                reference_frame,
                source_frames: stats.source_frames.clone(),
                primary_start: 0,
                primary_points: primary_count,
                completion_start: primary_count,
                completed_points: completion_count,
                sampled_pixels: stats.sampled_pixels,
                accepted_points: points.len(),
                reciprocal_consistent_points: stats.reciprocal_consistent_points,
                grid_stride: stats.grid_stride,
                grid_border: stats.grid_border,
                search_min_depth: stats.search_min_depth,
                search_max_depth: stats.search_max_depth,
            }]
        })
        .unwrap_or_default();

    DenseAnalysis {
        stats: DenseStats {
            attempted: stats.attempted,
            skip_reason: stats.skip_reason,
            reference_frame: stats.reference_frame,
            source_views: stats.source_views,
            source_frames: stats.source_frames,
            sampled_pixels: stats.sampled_pixels,
            depth_hypotheses: stats.depth_hypotheses,
            accepted_points: stats.accepted_points,
            surface_completion_proposals: stats.surface_completion_proposals,
            surface_completed_points: stats.surface_completed_points,
            surface_completion_rejected_texture: stats.surface_completion_rejected_texture,
            surface_completion_rejected_cross_view: stats.surface_completion_rejected_cross_view,
            surface_completion_rejected_reciprocal: stats.surface_completion_rejected_reciprocal,
            surface_completion_rejected_fusion: stats.surface_completion_rejected_fusion,
            surface_completion_rejected_footprint: stats.surface_completion_rejected_footprint,
            grid_stride: stats.grid_stride,
            grid_border: stats.grid_border,
            reciprocal_checked_points: stats.reciprocal_checked_points,
            reciprocal_rejected_points: stats.reciprocal_rejected_points,
            reciprocal_consistent_points: stats.reciprocal_consistent_points,
            fusion_input_observations: stats.fusion_input_observations,
            fusion_rejected_observations: stats.fusion_rejected_observations,
            fusion_rejected_points: stats.fusion_rejected_points,
            median_fusion_observations: stats.median_fusion_observations,
            median_supporting_views: stats.median_supporting_views,
            median_photometric_error: stats.median_photometric_error,
            search_min_depth: stats.search_min_depth,
            search_max_depth: stats.search_max_depth,
            reference_frames,
            reference_attempts,
            reference_patches,
        },
        points,
        grid_sites: convert_grid_sites(grid_sites),
    }
}

#[cfg(test)]
mod multi_reference_tests {
    use super::*;
    use nalgebra::Matrix3;

    fn camera(frame_index: usize, center_x: f64) -> RegisteredCamera {
        RegisteredCamera {
            frame_index,
            rotation: Matrix3::identity(),
            translation: Vector3::new(-center_x, 0.0, 0.0),
        }
    }

    fn texture(world_x: f64, world_y: f64) -> u8 {
        let value = 128.0
            + 55.0 * (world_x * 15.0 + world_y * 2.7).sin()
            + 40.0 * (world_y * 17.0 - world_x * 3.1).cos()
            + 25.0 * ((world_x + world_y) * 11.0).sin();
        value.round().clamp(0.0, 255.0) as u8
    }

    fn plane_frame(width: u32, height: u32, focal: f64, center_x: f64, depth: f64) -> FrameInput {
        let mut rgba = vec![0u8; width as usize * height as usize * 4];
        for y in 0..height {
            for x in 0..width {
                let world_x = center_x + (x as f64 - width as f64 * 0.5) / focal * depth;
                let world_y = (y as f64 - height as f64 * 0.5) / focal * depth;
                let value = texture(world_x, world_y);
                let index = (y as usize * width as usize + x as usize) * 4;
                rgba[index] = value;
                rgba[index + 1] = value;
                rgba[index + 2] = value;
                rgba[index + 3] = 255;
            }
        }
        FrameInput {
            width,
            height,
            rgba,
        }
    }

    fn sparse_plane(width: u32, height: u32, focal: f64, depth: f64) -> Vec<Vector3<f64>> {
        let mut points = Vec::new();
        for y in [10u32, 18, 26, 34] {
            for x in [12u32, 24, 36, 48, 56] {
                points.push(Vector3::new(
                    (x as f64 - width as f64 * 0.5) / focal * depth,
                    (y as f64 - height as f64 * 0.5) / focal * depth,
                    depth,
                ));
            }
        }
        points
    }

    #[test]
    fn reconstructs_multiple_reference_patches_without_relaxing_dense_gates() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let depth = 4.0;
        let frames = vec![
            plane_frame(width, height, focal, 0.0, depth),
            plane_frame(width, height, focal, 0.24, depth),
            plane_frame(width, height, focal, -0.22, depth),
        ];
        let cameras = vec![camera(0, 0.0), camera(1, 0.24), camera(2, -0.22)];
        let sparse = sparse_plane(width, height, focal, depth);

        let result = estimate_depth_points(&frames, &cameras, &sparse, focal);

        assert!(result.stats.attempted);
        assert!(result.stats.reference_frames.len() >= 2);
        assert_eq!(
            result.stats.reference_frames.len(),
            result.stats.reference_patches.len()
        );
        assert_eq!(
            result.stats.reference_patches.len(),
            result
                .stats
                .reference_attempts
                .iter()
                .filter(|attempt| attempt.accepted)
                .count()
        );
        assert!(result
            .stats
            .reference_attempts
            .iter()
            .all(|attempt| attempt.accepted || attempt.skip_reason.is_some()));
        assert_eq!(result.points.len(), result.grid_sites.len());
        assert_eq!(result.points.len(), result.stats.accepted_points);
        assert_eq!(
            result.stats.surface_completion_proposals,
            result.stats.surface_completed_points
                + result.stats.surface_completion_rejected_texture
                + result.stats.surface_completion_rejected_cross_view
                + result.stats.surface_completion_rejected_reciprocal
                + result.stats.surface_completion_rejected_fusion
                + result.stats.surface_completion_rejected_footprint
        );
        assert_eq!(
            result.stats.source_frames.len(),
            result.stats.source_views
        );
        assert!(!result
            .stats
            .source_frames
            .contains(&result.stats.reference_frame.expect("primary reference")));
    }

    #[test]
    fn rejected_reference_attempt_keeps_its_diagnostics() {
        let stats = legacy::DenseStats {
            attempted: true,
            sampled_pixels: 41,
            reciprocal_checked_points: 2,
            reciprocal_rejected_points: 1,
            reciprocal_consistent_points: 1,
            surface_completion_proposals: 3,
            surface_completed_points: 1,
            surface_completion_rejected_cross_view: 2,
            ..legacy::DenseStats::default()
        };

        let attempt = reference_attempt_stats(7, &stats, 2, false);

        assert!(attempt.attempted);
        assert!(!attempt.accepted);
        assert_eq!(attempt.reference_frame, 7);
        assert_eq!(attempt.sampled_pixels, 41);
        assert_eq!(attempt.accepted_points, 2);
        assert_eq!(attempt.reciprocal_checked_points, 2);
        assert_eq!(attempt.reciprocal_rejected_points, 1);
        assert_eq!(attempt.reciprocal_consistent_points, 1);
        assert_eq!(attempt.surface_completion_proposals, 3);
        assert_eq!(attempt.surface_completed_points, 1);
        assert_eq!(attempt.surface_completion_rejected_cross_view, 2);
        assert!(attempt
            .skip_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("at least 3")));
    }

    #[test]
    fn successful_fallback_evidence_replaces_rejected_attempt_for_same_reference() {
        let rejected = DenseReferenceAttemptStats {
            reference_frame: 3,
            attempted: true,
            accepted: false,
            skip_reason: Some("remapped reference rejected".into()),
            accepted_points: 1,
            ..DenseReferenceAttemptStats::default()
        };
        let other_rejected = DenseReferenceAttemptStats {
            reference_frame: 5,
            attempted: true,
            accepted: false,
            skip_reason: Some("another reference rejected".into()),
            ..DenseReferenceAttemptStats::default()
        };
        let fallback_accepted = DenseReferenceAttemptStats {
            reference_frame: 3,
            attempted: true,
            accepted: true,
            skip_reason: None,
            accepted_points: 9,
            ..DenseReferenceAttemptStats::default()
        };

        let merged = merge_fallback_attempts(
            vec![rejected, other_rejected],
            vec![fallback_accepted],
        );

        assert_eq!(merged.len(), 2);
        let accepted = merged
            .iter()
            .find(|attempt| attempt.reference_frame == 3)
            .expect("fallback reference evidence");
        assert!(accepted.accepted);
        assert_eq!(accepted.accepted_points, 9);
        assert!(accepted.skip_reason.is_none());
        assert!(merged
            .iter()
            .any(|attempt| attempt.reference_frame == 5 && !attempt.accepted));
    }
}
