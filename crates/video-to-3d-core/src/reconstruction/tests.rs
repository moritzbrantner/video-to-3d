use super::bootstrap::{diagnose_bootstrap, ineligible_link_gate};
use super::features::{detect_features, to_luma};
use super::matching::{compensate_global_motion, match_features, median, motion_guided_matches};
use super::recovery::{
    needs_registration_recovery, pan_recovery_radius, reconstruct, reconstruct_sparse,
    reconstruction_score, secondary_segment_solves,
};
use super::seed::{compose_track_matches, refine_subpixel, wide_seed_candidate};
use super::solve::{reconstruct_once, MIN_TRACK_MATCHES};
use super::types::{
    BootstrapEscalation, BootstrapGate, Feature, FeatureMatch, FrameInput, ReconstructionOptions,
    ReconstructionRequest, SeedCandidateStats, SeedGate, SeedGateCount, SegmentSolveStats,
    TrackScope,
};
use super::warnings::dense_warning;
use crate::{DenseStats, SegmentBreak};

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
fn composed_matches_follow_each_feature_through_every_link() {
    let link = |pairs: &[(usize, usize, f32)]| -> Vec<FeatureMatch> {
        pairs
            .iter()
            .map(|&(a, b, distance)| FeatureMatch { a, b, distance })
            .collect()
    };
    let composed = compose_track_matches(&[
        link(&[(0, 5, 0.1), (1, 6, 0.2), (2, 7, 0.1)]),
        link(&[(5, 9, 0.3), (7, 4, 0.05)]),
        link(&[(9, 1, 0.1), (4, 3, 0.2)]),
    ]);
    // Feature 1 is lost at the second link; distances are the worst link's.
    assert_eq!(
        composed
            .iter()
            .map(|m| (m.a, m.b, m.distance))
            .collect::<Vec<_>>(),
        vec![(0, 1, 0.3), (2, 3, 0.2)]
    );
    assert!(compose_track_matches(&[]).is_empty());
}

#[test]
fn wide_pairs_without_enough_refined_matches_stop_at_the_match_gate() {
    let (width, height) = (64_u32, 48_u32);
    let luma = vec![128_u8; (width * height) as usize];
    // Ten matches moving 3 px, all on the border where no patch fits.
    let feature = |x: u32, y: u32| Feature {
        x,
        y,
        score: 1.0,
        descriptor: vec![0],
    };
    let source: Vec<Feature> = (0..10).map(|index| feature(1, 4 + index * 4)).collect();
    let target: Vec<Feature> = (0..10).map(|index| feature(4, 4 + index * 4)).collect();
    let matches: Vec<FeatureMatch> = (0..10)
        .map(|index| FeatureMatch {
            a: index,
            b: index,
            distance: 0.1,
        })
        .collect();
    let (seed, estimate) = wide_seed_candidate(
        [&source, &target],
        [&luma, &luma],
        &matches,
        [0, 3],
        width,
        height,
        50.0,
    );
    assert!(estimate.is_none());
    assert_eq!(seed.rejected_gate, Some(SeedGate::Matches));
    assert_eq!(seed.rotation_only_residual_pixels, None);
}

#[test]
fn subpixel_refinement_recovers_a_fractional_shift() {
    let (width, height) = (48_u32, 40_u32);
    // A smooth texture, sampled once in place and once shifted by (0.3, -0.2) px.
    let render = |shift_x: f64, shift_y: f64| -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let (u, v) = (x as f64 - shift_x, y as f64 - shift_y);
                    (128.0
                        + 60.0 * (u * 0.45).sin() * (v * 0.38).cos()
                        + 40.0 * (u * 0.21 + v * 0.17).sin())
                    .round() as u8
                })
            })
            .collect()
    };
    let (source, target) = (render(0.0, 0.0), render(0.3, -0.2));
    let refined = refine_subpixel([&source, &target], width, height, [20, 18], [20, 18]).unwrap();
    assert!((refined[0] - 20.3).abs() < 0.08, "x {}", refined[0]);
    assert!((refined[1] - 17.8).abs() < 0.08, "y {}", refined[1]);
    // A whole-pixel offset inside the search window is found too.
    let refined = refine_subpixel([&source, &target], width, height, [20, 18], [19, 18]).unwrap();
    assert!((refined[0] - 20.3).abs() < 0.08, "x {}", refined[0]);
    // Too close to the border.
    assert_eq!(
        refine_subpixel([&source, &target], width, height, [2, 18], [2, 18]),
        None
    );
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
fn motion_guided_matching_recovers_displacement_beyond_local_radius() {
    let source: Vec<Feature> = (0u32..8)
        .map(|index| Feature {
            x: 10 + index * 8,
            y: 16 + (index % 3) * 11,
            score: 10.0 - index as f32 * 0.1,
            descriptor: vec![index as i16 * 32, index as i16 * 32 + 3],
        })
        .collect();
    let target: Vec<Feature> = source
        .iter()
        .map(|feature| Feature {
            x: feature.x + 48,
            y: feature.y + 2,
            score: feature.score,
            descriptor: feature.descriptor.clone(),
        })
        .collect();
    let options = ReconstructionOptions {
        match_radius: 12,
        max_descriptor_distance: 6.0,
        ratio_threshold: 0.8,
        ..ReconstructionOptions::default()
    };
    assert!(match_features(&source, &target, options).is_empty());
    let guided = motion_guided_matches(&source, &target, options)
        .expect("coarse displacement should center the bounded search");
    assert_eq!(guided.len(), source.len());
    let mut dx: Vec<f32> = guided
        .iter()
        .map(|feature_match| target[feature_match.b].x as f32 - source[feature_match.a].x as f32)
        .collect();
    let mut dy: Vec<f32> = guided
        .iter()
        .map(|feature_match| target[feature_match.b].y as f32 - source[feature_match.a].y as f32)
        .collect();
    assert_eq!(median(&mut dx), 48.0);
    assert_eq!(median(&mut dy), 2.0);
}

#[test]
fn recovery_radius_remains_bounded_and_tracks_observed_motion() {
    let request = ReconstructionRequest {
        frames: vec![synthetic_frame(96, 80, 0), synthetic_frame(96, 80, 0)],
        options: ReconstructionOptions::default(),
    };
    let mut result = reconstruct_once(&request).expect("fixture should reconstruct");
    assert_eq!(pan_recovery_radius(360, 42, &result), 108);
    assert_eq!(pan_recovery_radius(180, 42, &result), 54);
    assert_eq!(pan_recovery_radius(96, 42, &result), 42);
    assert_eq!(pan_recovery_radius(360, 128, &result), 128);
    result.pairs[0].median_motion = 44.0;
    assert_eq!(pan_recovery_radius(240, 42, &result), 104);
}

#[test]
fn recovery_score_is_independent_of_dense_budget_output() {
    // Recovery ranks the camera stage's sparse solve, which carries no dense depth or
    // mesh; the dense working-set budget only reaches the later dense stage.
    let request = ReconstructionRequest {
        frames: vec![synthetic_frame(96, 80, 0), synthetic_frame(96, 80, 0)],
        options: ReconstructionOptions::default(),
    };
    let mut starved = request.clone();
    starved.options.max_dense_working_set_bytes = 1;
    let ordinary = reconstruct_once(&request).expect("fixture should reconstruct");
    let mut budgeted = reconstruct_once(&starved).expect("fixture should reconstruct");

    assert_eq!(
        reconstruction_score(&ordinary, ordinary.track_scope.primary),
        reconstruction_score(&budgeted, budgeted.track_scope.primary)
    );
    budgeted.options.max_dense_working_set_bytes = request.options.max_dense_working_set_bytes;
    assert_eq!(
        budgeted, ordinary,
        "the budget changes no camera-stage output"
    );
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
    assert!(!warning.contains("meshing"));
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

const SCENE_WIDTH: u32 = 640;
const SCENE_HEIGHT: u32 = 480;
const SCENE_FOCAL: f64 = 520.0;

/// Textured blobs scattered through a box in front of the camera, from a fixed seed.
fn scene_points(seed: u64) -> Vec<(nalgebra::Vector3<f64>, u8)> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    (0..420)
        .map(|index| {
            let point = nalgebra::Vector3::new(
                (next() - 0.5) * 5.2,
                (next() - 0.5) * 3.5,
                4.0 + next() * 6.0,
            );
            (point, 35 + ((index * 47) % 190) as u8)
        })
        .collect()
}

/// One frame of a lateral camera move through `points`, on a flat `background`.
fn render_scene(
    points: &[(nalgebra::Vector3<f64>, u8)],
    camera_x: f64,
    background: u8,
) -> FrameInput {
    render_scene_posed(points, camera_x, 0.0, background)
}

/// One frame from a camera at `camera_x` turned by `yaw` radians about the vertical axis.
fn render_scene_posed(
    points: &[(nalgebra::Vector3<f64>, u8)],
    camera_x: f64,
    yaw: f64,
    background: u8,
) -> FrameInput {
    let (width, height) = (SCENE_WIDTH, SCENE_HEIGHT);
    let mut rgba = vec![background; (width * height * 4) as usize];
    for pixel in rgba.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    let (sin, cos) = yaw.sin_cos();
    for (index, (point, shade)) in points.iter().enumerate() {
        let (x, z) = (point.x - camera_x, point.z);
        let (x, z) = (cos * x - sin * z, sin * x + cos * z);
        if z <= 0.5 {
            continue;
        }
        let px = (x / z * SCENE_FOCAL + width as f64 * 0.5).round() as i32;
        let py = (point.y / z * SCENE_FOCAL + height as f64 * 0.5).round() as i32;
        let radius = 3 + (index % 2) as i32;
        if px < radius + 1
            || py < radius + 1
            || px >= width as i32 - radius - 1
            || py >= height as i32 - radius - 1
        {
            continue;
        }
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let offset = (((py + dy) as u32 * width + (px + dx) as u32) * 4) as usize;
                let value = if ((dx + dy + index as i32) & 1) == 0 {
                    *shade
                } else {
                    255_u8.saturating_sub(shade / 2)
                };
                rgba[offset] = value;
                rgba[offset + 1] = value.saturating_add((index % 17) as u8);
                rgba[offset + 2] = value.saturating_sub((index % 13) as u8);
            }
        }
    }
    FrameInput {
        width,
        height,
        rgba,
    }
}

fn lateral_span(points: &[(nalgebra::Vector3<f64>, u8)], background: u8) -> Vec<FrameInput> {
    (0..6)
        .map(|frame| render_scene(points, -0.42 + frame as f64 * 0.12, background))
        .collect()
}

fn scene_request(frames: Vec<FrameInput>) -> ReconstructionRequest {
    ReconstructionRequest {
        frames,
        options: ReconstructionOptions {
            min_feature_distance: 6,
            match_radius: 80,
            focal_length_pixels: Some(SCENE_FOCAL as f32),
            ..ReconstructionOptions::default()
        },
    }
}

#[test]
fn hard_cut_clip_solves_each_segment_from_its_own_seed() {
    let mut frames = lateral_span(&scene_points(0x5eed), 236);
    frames.extend(lateral_span(&scene_points(0xc0ffee), 64));

    let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");
    let segments = &result.multi_view.keyframe_selection.segments;
    assert_eq!(segments.len(), 2, "segments: {segments:?}");
    assert_eq!(segments[0].ends_with, Some(SegmentBreak::HardCut));

    let solves = &result.multi_view.segment_solves;
    assert_eq!(solves.len(), 2);
    assert_eq!(solves.iter().filter(|solve| solve.primary).count(), 1);
    for (segment, solve) in solves.iter().enumerate() {
        assert_eq!(solve.segment, segment);
        let span = &segments[segment];
        let seed = solve
            .seed_pair
            .as_ref()
            .unwrap_or_else(|| panic!("segment {segment} has no seed: {solve:?}"));
        assert!(seed.from_frame >= span.first_frame && seed.to_frame <= span.last_frame);
        assert!(
            solve.registered_views > 0,
            "segment {segment} registered no views: {solve:?}"
        );
        assert_eq!(solve.cameras.len(), 2 + solve.registered_views);
        // Neither camera set reaches into the other segment's frames.
        assert!(solve.cameras.iter().all(|camera| {
            camera.frame_index >= span.first_frame && camera.frame_index <= span.last_frame
        }));
    }

    // The primary solve is the top-level result.
    let primary = solves.iter().find(|solve| solve.primary).unwrap();
    let calibrated = result.calibrated_pair.as_ref().unwrap();
    assert_eq!(
        primary.seed_pair.as_ref().unwrap().from_frame,
        calibrated.from_frame
    );
    assert_eq!(primary.registered_views, result.registered_views.len());
    assert_eq!(
        primary
            .cameras
            .iter()
            .map(|camera| camera.frame_index)
            .collect::<Vec<_>>(),
        result
            .cameras
            .iter()
            .map(|camera| camera.frame_index)
            .collect::<Vec<_>>()
    );
}

#[test]
fn single_segment_clip_reports_no_segment_solves() {
    let frames = lateral_span(&scene_points(0x5eed), 236);
    let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");

    assert_eq!(result.multi_view.keyframe_selection.segments.len(), 1);
    assert!(result.calibrated_pair.is_some());
    assert!(result.multi_view.segment_solves.is_empty());
    let serialized = serde_json::to_string(&result.multi_view).unwrap();
    assert!(!serialized.contains("segment_solves"));
}

#[test]
fn accepted_lateral_clip_records_its_seed_candidates() {
    let frames = lateral_span(&scene_points(0x5eed), 236);
    let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");

    let pair = result.calibrated_pair.as_ref().expect("lateral clip seeds");
    assert!(!result.registered_views.is_empty());
    assert_eq!(result.seed_candidates.len(), 5);
    let seed = &result.seed_candidates[pair.from_frame];
    assert!(seed.selected && seed.rejected_gate.is_none());
    assert_eq!(seed.inliers, Some(pair.inliers));
    assert!(seed.rotation_only_residual_pixels.unwrap() > 1.25);
    assert_eq!(result.bootstrap.decisive_gate, None);
    assert_eq!(result.bootstrap.escalation, None);
    assert_eq!(
        result
            .seed_candidates
            .iter()
            .filter(|seed| seed.selected)
            .count(),
        1
    );
}

/// The Trevi canary's measured limitation in isolation: a camera that only turns.
/// Every adjacent pair has abundant consistent matches, but a pure rotation explains
/// them, so no seed pair and therefore no camera may be accepted.
#[test]
fn pure_pan_clip_names_the_baseline_gate_instead_of_inventing_cameras() {
    let points = scene_points(0x7e71);
    let frames = (0..6)
        .map(|frame| render_scene_posed(&points, 0.0, frame as f64 * 0.02, 236))
        .collect();
    let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");

    assert!(result.calibrated_pair.is_none());
    assert!(result.registered_views.is_empty());
    assert!(!result.dense.attempted);
    // Every adjacent pair, and every wider pair tried because none of them seeded,
    // is rejected.
    let adjacent = result
        .seed_candidates
        .iter()
        .filter(|seed| seed.to_frame == seed.from_frame + 1)
        .count();
    assert_eq!(adjacent, 5);
    assert!(result.seed_candidates.len() > adjacent);
    for seed in &result.seed_candidates {
        assert!(!seed.selected);
        assert!(seed.rejected_gate.is_some(), "accepted pan pair: {seed:?}");
    }
    assert_eq!(
        result.bootstrap.decisive_gate,
        Some(BootstrapGate::Seed(SeedGate::Baseline)),
        "bootstrap: {:?}",
        result.bootstrap
    );
    assert_eq!(
        result.bootstrap.escalation,
        Some(BootstrapEscalation::LearnedMultiView)
    );
    let decisive = result.bootstrap.decisive_pair.as_ref().unwrap();
    assert!(decisive.inliers.unwrap() >= 8);
    assert!(decisive.rotation_only_residual_pixels.unwrap() <= 1.25);
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning.contains("decisive gate is translation baseline")));

    let serialized = serde_json::to_value(&result.bootstrap).unwrap();
    assert_eq!(serialized["decisive_gate"], "baseline");
    assert_eq!(serialized["escalation"], "learned_multi_view");
}

#[test]
fn slow_lateral_steps_seed_from_a_wider_pair() {
    // Steps too small for any adjacent pair to show residual parallax.
    let points = scene_points(0x5107);
    let frames = (0..8)
        .map(|frame| render_scene(&points, -0.1 + frame as f64 * 0.03, 236))
        .collect();
    let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");

    for seed in result
        .seed_candidates
        .iter()
        .filter(|seed| seed.to_frame == seed.from_frame + 1)
    {
        assert!(
            seed.rejected_gate.is_some(),
            "adjacent pair seeded: {seed:?}"
        );
    }
    let pair = result.calibrated_pair.as_ref().expect("a wider pair seeds");
    assert!(pair.to_frame > pair.from_frame + 1, "{pair:?}");
    let selected = result
        .seed_candidates
        .iter()
        .find(|seed| seed.selected)
        .expect("the seed is recorded");
    assert_eq!(
        (selected.from_frame, selected.to_frame),
        (pair.from_frame, pair.to_frame)
    );
    assert!(selected.rotation_only_residual_pixels.unwrap() > 1.25);
    assert_eq!(result.bootstrap.decisive_gate, None);
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning.contains("sampling intervals")));
    assert!(!result.registered_views.is_empty());
}

fn seed_stats(
    from_frame: usize,
    gate: Option<SeedGate>,
    inliers: Option<usize>,
) -> SeedCandidateStats {
    SeedCandidateStats {
        from_frame,
        to_frame: from_frame + 1,
        matches: 40,
        median_motion: 6.0,
        median_parallax_residual: 1.0,
        rejected_gate: gate,
        selected: false,
        inliers,
        inlier_ratio: inliers.map(|inliers| inliers as f32 / 40.0),
        rotation_only_residual_pixels: None,
        required_points: None,
        triangulated_points: None,
        behind_camera: None,
        failed_triangulations: None,
        high_reprojection: None,
        median_reprojection_error_pixels: None,
        median_triangulation_angle_degrees: None,
    }
}

#[test]
fn bootstrap_names_the_gate_of_the_furthest_candidate() {
    let seeds = vec![
        seed_stats(0, Some(SeedGate::Parallax), None),
        seed_stats(1, Some(SeedGate::Baseline), Some(30)),
        seed_stats(2, Some(SeedGate::TriangulationAngle), Some(20)),
        seed_stats(3, Some(SeedGate::Baseline), Some(36)),
        seed_stats(4, Some(SeedGate::TriangulationAngle), Some(25)),
    ];
    let diagnosis = diagnose_bootstrap(&seeds, false, false, 0, 0);
    // The furthest stage wins over more inliers at an earlier gate; inliers break ties.
    assert_eq!(
        diagnosis.decisive_gate,
        Some(BootstrapGate::Seed(SeedGate::TriangulationAngle))
    );
    assert_eq!(diagnosis.decisive_pair.as_ref().unwrap().from_frame, 4);
    assert_eq!(
        diagnosis.escalation,
        Some(BootstrapEscalation::LearnedMultiView)
    );
    assert_eq!(
        diagnosis.rejections,
        vec![
            SeedGateCount {
                gate: SeedGate::Parallax,
                candidates: 1
            },
            SeedGateCount {
                gate: SeedGate::Baseline,
                candidates: 2
            },
            SeedGateCount {
                gate: SeedGate::TriangulationAngle,
                candidates: 2
            },
        ]
    );
    assert!(
        diagnosis.summary.contains("frames 5–6"),
        "{}",
        diagnosis.summary
    );

    let correspondence_limited = diagnose_bootstrap(
        &[seed_stats(0, Some(SeedGate::EssentialInliers), Some(9))],
        false,
        false,
        0,
        0,
    );
    assert_eq!(
        correspondence_limited.escalation,
        Some(BootstrapEscalation::LearnedMatching)
    );
    let unusable = diagnose_bootstrap(
        &[seed_stats(0, Some(SeedGate::FrameQuality), None)],
        false,
        false,
        0,
        0,
    );
    assert_eq!(unusable.escalation, None);
}

#[test]
fn too_few_matches_without_a_cut_report_the_match_gate() {
    assert_eq!(
        ineligible_link_gate(Some(SegmentBreak::LostOverlap), MIN_TRACK_MATCHES - 1),
        SeedGate::Matches
    );
    assert_eq!(
        ineligible_link_gate(Some(SegmentBreak::LostOverlap), 0),
        SeedGate::Matches
    );
    // A hard cut, a motion jump, or lost overlap with enough matches is a segment break.
    assert_eq!(
        ineligible_link_gate(Some(SegmentBreak::HardCut), 0),
        SeedGate::SegmentBreak
    );
    assert_eq!(
        ineligible_link_gate(Some(SegmentBreak::MotionJump), 40),
        SeedGate::SegmentBreak
    );
    assert_eq!(
        ineligible_link_gate(Some(SegmentBreak::LostOverlap), MIN_TRACK_MATCHES),
        SeedGate::SegmentBreak
    );
}

#[test]
fn bootstrap_reports_registration_after_an_accepted_seed() {
    let seeds = vec![
        seed_stats(0, None, Some(40)),
        seed_stats(1, Some(SeedGate::Baseline), Some(30)),
    ];
    let stalled = diagnose_bootstrap(&seeds, true, false, 0, 3);
    assert_eq!(stalled.decisive_gate, Some(BootstrapGate::Registration));
    assert!(stalled.decisive_pair.is_none());
    assert!(stalled.summary.contains("3 other selected keyframe(s)"));
    assert_eq!(
        serde_json::to_value(stalled.decisive_gate).unwrap(),
        "registration"
    );

    let registered = diagnose_bootstrap(&seeds, true, false, 2, 3);
    assert_eq!(registered.decisive_gate, None);
    assert_eq!(registered.escalation, None);

    // No keyframe reached PnP: the limit is correspondence, not a rejected pose.
    let untried = diagnose_bootstrap(&seeds, true, false, 0, 0);
    assert_eq!(untried.decisive_gate, Some(BootstrapGate::Registration));
    assert!(
        untried.summary.contains("no registration was tried"),
        "{}",
        untried.summary
    );
    assert!(!untried.summary.contains("inlier and reprojection gates"));

    // A two-frame clip is fully covered by its seed pair.
    let two_frames = diagnose_bootstrap(&seeds[..1], true, true, 0, 0);
    assert_eq!(two_frames.decisive_gate, None);
    assert_eq!(two_frames.escalation, None);
}

/// A split clip whose second segment pans too far for the first pass to seed it.
fn wide_pan_split_request() -> ReconstructionRequest {
    let mut frames = lateral_span(&scene_points(0x5eed), 236);
    let wide = scene_points(0xc0ffee);
    frames.extend((0..6).map(|frame| render_scene(&wide, -0.42 + frame as f64 * 1.2, 64)));
    scene_request(frames)
}

#[test]
fn recovery_keeps_a_secondary_segment_only_the_retry_solves() {
    let request = wide_pan_split_request();
    let initial = reconstruct_once(&request).expect("first pass should succeed");
    let initial_secondary: Vec<_> = secondary_segment_solves(&initial).collect();
    assert_eq!(initial_secondary.len(), 1);
    assert!(initial_secondary[0].seed_pair.is_none());

    let result = reconstruct_sparse(&request).expect("reconstruction should succeed");
    let secondary: Vec<_> = secondary_segment_solves(&result).collect();
    assert_eq!(secondary.len(), 1);
    assert!(secondary[0].seed_pair.is_some(), "{:?}", secondary[0]);
    assert!(secondary[0].registered_views > 0);
    assert!(
        reconstruction_score(&result, initial.track_scope.primary)
            > reconstruction_score(&initial, initial.track_scope.primary)
    );
}

#[test]
fn track_scope_counts_only_frames_inside_the_scope() {
    let single = reconstruct_once(&scene_request(lateral_span(&scene_points(0x5eed), 236)))
        .expect("single segment should reconstruct");
    let stats = &single.multi_view;
    assert_eq!(single.track_scope.primary, (0, 5));
    assert_eq!(
        single.track_scope.stats(single.track_scope.primary),
        [
            stats.tracks_three_plus,
            stats.longest_track,
            stats.linked_pairs
        ]
    );

    let split = reconstruct_once(&wide_pan_split_request()).expect("split should reconstruct");
    let stats = &split.multi_view;
    let primary = split.track_scope.primary;
    assert!(stats.keyframe_selection.segments.len() > 1);
    assert!(
        primary.1 - primary.0 < 11,
        "the primary scope is one segment"
    );
    let [three_plus, longest, linked] = split.track_scope.stats(primary);
    assert!(
        linked < stats.linked_pairs,
        "the other segment's pairs are excluded"
    );
    assert!(three_plus <= stats.tracks_three_plus && longest <= stats.longest_track);
    // Over the whole clip, the scope reproduces the clip-wide statistics.
    assert_eq!(
        split.track_scope.stats((0, 11)),
        [
            stats.tracks_three_plus,
            stats.longest_track,
            stats.linked_pairs
        ]
    );

    let scope = TrackScope {
        spans: vec![(0, 4), (3, 5), (6, 6)],
        linked: vec![true, true, false, true, true],
        primary: (0, 5),
    };
    // Spans observe 5, 3 and 1 frames; within 2..=4 they observe 3, 2 and 0.
    assert_eq!(scope.stats((0, 5)), [2, 5, 4]);
    assert_eq!(scope.stats((2, 4)), [1, 3, 1]);
}

#[test]
fn a_short_secondary_segment_alone_triggers_recovery() {
    let request = scene_request(lateral_span(&scene_points(0x5eed), 236));
    let mut result = reconstruct_once(&request).expect("fixture should reconstruct");
    assert!(!needs_registration_recovery(&request, &result));

    let solve = SegmentSolveStats {
        segment: 1,
        first_frame: 6,
        last_frame: 11,
        primary: false,
        seed_pair: result.calibrated_pair.clone(),
        registration_candidates: 3,
        pnp_ready_candidates: 3,
        registered_views: 3,
        recovered_from_revisit: 0,
        sparse_points: 100,
        bundle_adjustment_accepted: true,
        cameras: Vec::new(),
    };
    result.multi_view.segment_solves = vec![solve.clone()];
    assert!(!needs_registration_recovery(&request, &result));

    result.multi_view.segment_solves[0].registered_views = 2;
    assert!(needs_registration_recovery(&request, &result));
    result.multi_view.segment_solves[0] = SegmentSolveStats {
        seed_pair: None,
        ..solve.clone()
    };
    assert!(needs_registration_recovery(&request, &result));

    // A one-frame segment cannot be seeded, so it never asks for a retry.
    result.multi_view.segment_solves[0] = SegmentSolveStats {
        seed_pair: None,
        first_frame: 6,
        last_frame: 6,
        registration_candidates: 0,
        pnp_ready_candidates: 0,
        registered_views: 0,
        ..solve.clone()
    };
    assert!(!needs_registration_recovery(&request, &result));

    // A revisit-recovered view does not cover a PnP-ready candidate that failed.
    result.multi_view.segment_solves[0] = SegmentSolveStats {
        registered_views: 3,
        recovered_from_revisit: 1,
        ..solve
    };
    assert!(needs_registration_recovery(&request, &result));
}

#[test]
fn secondary_segments_never_outrank_the_primary_solve() {
    let request = scene_request(lateral_span(&scene_points(0x5eed), 236));
    let base = reconstruct_once(&request).expect("fixture should reconstruct");
    let solve = |seeded: bool, registered_views: usize| SegmentSolveStats {
        segment: 1,
        first_frame: 6,
        last_frame: 11,
        primary: false,
        seed_pair: base.calibrated_pair.clone().filter(|_| seeded),
        registration_candidates: 4,
        pnp_ready_candidates: 4,
        registered_views,
        recovered_from_revisit: 0,
        sparse_points: registered_views * 50,
        bundle_adjustment_accepted: seeded,
        cameras: if seeded {
            base.cameras
                .iter()
                .copied()
                .take(2 + registered_views)
                .collect()
        } else {
            Vec::new()
        },
    };
    let mut unseeded = base.clone();
    unseeded.multi_view.segment_solves = vec![solve(false, 0)];
    let mut seeded = base.clone();
    seeded.multi_view.segment_solves = vec![solve(true, 3)];
    assert!(
        reconstruction_score(&seeded, seeded.track_scope.primary)
            > reconstruction_score(&unseeded, unseeded.track_scope.primary)
    );

    let mut stronger_primary = unseeded.clone();
    stronger_primary.points.push(base.points[0]);
    assert!(
        reconstruction_score(&stronger_primary, stronger_primary.track_scope.primary)
            > reconstruction_score(&seeded, seeded.track_scope.primary)
    );

    // Ranking reads the primary segment's track statistics, not the clip-wide ones.
    let mut clip_wide = seeded.clone();
    clip_wide.multi_view.tracks_three_plus += 100;
    clip_wide.multi_view.longest_track += 10;
    clip_wide.multi_view.linked_pairs += 10;
    assert_eq!(
        reconstruction_score(&clip_wide, clip_wide.track_scope.primary),
        reconstruction_score(&seeded, seeded.track_scope.primary)
    );

    // Two seeded fragments with fewer cameras lose to one segment with more.
    let mut fragmented = base.clone();
    fragmented.multi_view.segment_solves = vec![solve(true, 0), solve(true, 0)];
    let mut merged = base.clone();
    merged.multi_view.segment_solves = vec![solve(true, 3)];
    let scope = base.track_scope.primary;
    assert!(reconstruction_score(&merged, scope) > reconstruction_score(&fragmented, scope));
}
