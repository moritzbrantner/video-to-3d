//! One deterministic camera solve over a whole clip.

use super::bootstrap::{diagnose_bootstrap, ineligible_link_gate};
use super::control::ReconstructionControl;
use super::features::{detect_features_with_control, sample_rgb, to_luma};
use super::matching::{compensate_global_motion, match_features, median, motion_guided_matches};
use super::seed::{
    best_seed_index, calibrated_pair_stats, compose_track_matches, record_two_view_evidence,
    seed_pair_cameras, wide_seed_candidate_with_control, MAX_SEED_SPAN_INTERVALS,
};
use super::segment::{segment_solve_stats, solve_segment, SegmentSolve};
use super::types::{
    CameraPose, PairStats, Point3, ReconstructionRequest, SeedCandidateStats, SeedGate,
    SegmentSolveStats, TrackScope,
};
use crate::reconstruction_stages::{SparseReconstruction, StageCamera};
use crate::{keyframe_selection, revisit, two_view};
use std::collections::HashMap;

pub(crate) const MIN_TRACK_MATCHES: usize = 8;
pub(crate) const MIN_TRACK_OVERLAP: f32 = 0.18;

/// One deterministic camera solve with the request's options: the sparse result and the
/// camera-stage diagnostics, but no dense depth or mesh (those are later stages).
#[cfg(test)]
pub(crate) fn reconstruct_once(
    request: &ReconstructionRequest,
) -> Result<SparseReconstruction, String> {
    reconstruct_once_with_control(request, &mut ReconstructionControl::new(&mut || false))
}

pub(super) fn reconstruct_once_with_control(
    request: &ReconstructionRequest,
    control: &mut ReconstructionControl<'_>,
) -> Result<SparseReconstruction, String> {
    validate_request(request)?;

    let width = request.frames[0].width;
    let height = request.frames[0].height;
    let options = request.options;
    let mut luma_frames = Vec::with_capacity(request.frames.len());
    for frame in &request.frames {
        luma_frames.push(to_luma(frame));
        control.check()?;
    }
    let mut features = Vec::with_capacity(luma_frames.len());
    for luma in &luma_frames {
        features.push(detect_features_with_control(
            luma, width, height, options, control,
        )?);
    }

    let focal = options
        .focal_length_pixels
        .unwrap_or(0.86 * width.max(height) as f32);
    let diagonal = (width as f32).hypot(height as f32);
    let mut cameras = Vec::with_capacity(request.frames.len());
    let mut points = Vec::new();
    let mut pairs = Vec::with_capacity(request.frames.len() - 1);
    let mut adjacent_matches = Vec::with_capacity(request.frames.len() - 1);
    let mut warnings = Vec::new();
    let mut motion_guided_pairs = 0usize;
    // Every calibrated pair that passes the seed gates, in pair order. Each segment seeds
    // from its own best candidate.
    control.check()?;
    let mut seed_candidates: Vec<([usize; 2], two_view::TwoViewEstimate)> = Vec::new();
    // Whether each adjacent link may carry a seed: both frames usable and the link
    // seed-eligible. A wider seed pair needs every link between its frames.
    let mut seed_links: Vec<bool> = Vec::with_capacity(request.frames.len() - 1);
    let mut seed_evidence: Vec<SeedCandidateStats> = Vec::with_capacity(request.frames.len() - 1);
    let frame_evidence: Vec<_> = luma_frames
        .iter()
        .enumerate()
        .map(|(index, luma)| crate::input_readiness::frame_evidence(index, luma, width, height))
        .collect();
    let mut segmentation =
        keyframe_selection::SegmentationContext::new(&frame_evidence, width, height);
    for (before, after) in segmentation.rejected_runs() {
        control.check()?;
        let (source, target) = (&features[before], &features[after]);
        let mut matches = match_features(source, target, options);
        if let Some(guided) = motion_guided_matches(source, target, options) {
            if guided.len() > matches.len() {
                matches = guided;
            }
        }
        let overlap = if source.is_empty() || target.is_empty() {
            0.0
        } else {
            matches.len() as f32 / source.len().min(target.len()) as f32
        };
        let mut motion: Vec<f32> = matches
            .iter()
            .map(|m| {
                let (a, b) = (&source[m.a], &target[m.b]);
                (b.x as f32 - a.x as f32).hypot(b.y as f32 - a.y as f32)
            })
            .collect();
        let median_motion = median(&mut motion);
        segmentation.bridge(before, after, matches.len(), overlap, median_motion);
    }

    let mut camera = CameraPose {
        frame_index: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        matched_features: 0,
    };
    cameras.push(camera);

    for pair_index in 0..request.frames.len() - 1 {
        control.check()?;
        let source_features = &features[pair_index];
        let target_features = &features[pair_index + 1];
        let ordinary_matches = match_features(source_features, target_features, options);
        let ordinary_overlap = if source_features.is_empty() || target_features.is_empty() {
            0.0
        } else {
            ordinary_matches.len() as f32 / source_features.len().min(target_features.len()) as f32
        };
        let (matches, used_motion_guidance) =
            if ordinary_matches.len() < MIN_TRACK_MATCHES || ordinary_overlap < MIN_TRACK_OVERLAP {
                match motion_guided_matches(source_features, target_features, options) {
                    Some(guided) if guided.len() > ordinary_matches.len() => (guided, true),
                    _ => (ordinary_matches, false),
                }
            } else {
                (ordinary_matches, false)
            };
        if used_motion_guidance {
            motion_guided_pairs += 1;
        }
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

        // The seed uses two frames that pass the quality gates, within one segment.
        let seed_eligible =
            segmentation.seed_eligible(pair_index, matches.len(), overlap_ratio, median_motion);
        let mut seed = SeedCandidateStats {
            from_frame: pair_index,
            to_frame: pair_index + 1,
            matches: matches.len(),
            median_motion,
            median_parallax_residual,
            rejected_gate: None,
            selected: false,
            inliers: None,
            inlier_ratio: None,
            rotation_only_residual_pixels: None,
            required_points: None,
            triangulated_points: None,
            behind_camera: None,
            failed_triangulations: None,
            high_reprojection: None,
            median_reprojection_error_pixels: None,
            median_triangulation_angle_degrees: None,
        };
        seed_links.push(
            segmentation.usable(pair_index) && segmentation.usable(pair_index + 1) && seed_eligible,
        );
        if !segmentation.usable(pair_index) || !segmentation.usable(pair_index + 1) {
            seed.rejected_gate = Some(SeedGate::FrameQuality);
        } else if !seed_eligible {
            seed.rejected_gate = Some(ineligible_link_gate(
                segmentation.link_break(pair_index, matches.len(), overlap_ratio, median_motion),
                matches.len(),
            ));
        } else if low_parallax {
            seed.rejected_gate = Some(SeedGate::Parallax);
        } else {
            let outcome = two_view::evaluate_two_view_with_control(
                source_features,
                target_features,
                &matches,
                width,
                height,
                focal as f64,
                control,
            );
            control.check()?;
            record_two_view_evidence(&mut seed, &outcome.evidence);
            match outcome.result {
                Ok(estimate) => seed_candidates.push(([pair_index, pair_index + 1], estimate)),
                Err(rejection) => seed.rejected_gate = Some(SeedGate::from_two_view(rejection)),
            }
        }
        seed_evidence.push(seed);

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

    let mut keyframe_selection = keyframe_selection::select(&pairs, &frame_evidence, &segmentation);
    let revisit_context =
        revisit::RevisitContext::new(&features, width, height, focal as f64, options);
    let frame_segments: Vec<usize> = keyframe_selection
        .stats
        .frames
        .iter()
        .map(|frame| frame.segment)
        .collect();
    let segment_of_pair = |pair_index: usize| frame_segments[pair_index];
    let mut wide_seed_segments = Vec::new();
    // A slow lateral step gives each adjacent pair too little parallax to seed. A segment
    // without an adjacent seed tries wider pairs of the same segment, composing matches
    // along the tracked links, under the same fail-closed gates (#127).
    for (segment, span) in keyframe_selection.stats.segments.iter().enumerate() {
        control.check()?;
        if seed_candidates
            .iter()
            .any(|(frames, _)| segment_of_pair(frames[0]) == segment)
        {
            continue;
        }
        // Every span is tried; the widest-angle accepted pair seeds, since a short
        // low-parallax baseline gives an ill-conditioned essential matrix.
        let mut best: Option<([usize; 2], two_view::TwoViewEstimate)> = None;
        for interval in 2..=MAX_SEED_SPAN_INTERVALS {
            for from in span.first_frame..=span.last_frame.saturating_sub(interval) {
                control.check()?;
                let to = from + interval;
                if to > span.last_frame || !seed_links[from..to].iter().all(|usable| *usable) {
                    continue;
                }
                let (seed, estimate) = wide_seed_candidate_with_control(
                    [&features[from], &features[to]],
                    [&luma_frames[from], &luma_frames[to]],
                    &compose_track_matches(&adjacent_matches[from..to]),
                    [from, to],
                    width,
                    height,
                    focal,
                    control,
                );
                seed_evidence.push(seed);
                if let Some(estimate) = estimate {
                    if best.as_ref().is_none_or(|(_, current)| {
                        estimate.median_triangulation_angle_degrees
                            > current.median_triangulation_angle_degrees
                    }) {
                        best = Some(([from, to], estimate));
                    }
                }
            }
        }
        if let Some(best) = best {
            seed_candidates.push(best);
            wide_seed_segments.push(segment);
        }
    }
    let promoted_keyframes: HashMap<usize, usize> = wide_seed_segments
        .into_iter()
        .map(|segment| (segment, keyframe_selection.promote_segment_frames(segment)))
        .collect();
    let clip_segments = &keyframe_selection.stats.segments;
    // A split clip solves every segment that has a usable seed on its own, in its own
    // arbitrary frame. The segment whose solve registers the most cameras fills the
    // primary output (#230); the strongest seed breaks ties.
    let segment_seed = |segment: usize| {
        best_seed_index(&seed_candidates, |pair_index| {
            segment_of_pair(pair_index) == segment
        })
    };
    let mut secondary: Vec<(usize, Option<usize>, Option<SegmentSolve>)> = Vec::new();
    if clip_segments.len() > 1 {
        for segment in 0..clip_segments.len() {
            let seed = segment_seed(segment);
            let solve = seed
                .map(|index| {
                    solve_segment(
                        Some(&seed_candidates[index]),
                        &keyframe_selection,
                        &adjacent_matches,
                        &features,
                        &revisit_context,
                        width,
                        height,
                        focal as f64,
                        true,
                        control,
                    )
                })
                .transpose()?;
            secondary.push((segment, seed, solve));
        }
    }
    let strongest = best_seed_index(&seed_candidates, |_| true);
    let primary_index = if clip_segments.len() > 1 {
        secondary
            .iter()
            .filter_map(|(_, seed, solve)| {
                Some(((*seed)?, solve.as_ref()?.registered_geometry.len()))
            })
            .fold(
                None::<(usize, usize)>,
                |best, (seed, registered)| match best {
                    Some((best_seed, best_registered))
                        if best_registered > registered
                            || (best_registered == registered
                                && !seed_candidates[seed]
                                    .1
                                    .is_better_than(&seed_candidates[best_seed].1)) =>
                    {
                        Some((best_seed, best_registered))
                    }
                    _ => Some((seed, registered)),
                },
            )
            .map(|(seed, _)| seed)
            .or(strongest)
    } else {
        strongest
    };
    let primary_segment = primary_index.map(|index| segment_of_pair(seed_candidates[index].0[0]));
    let mut segment_solves: Vec<SegmentSolveStats> = secondary
        .iter()
        .filter(|(segment, _, _)| Some(*segment) != primary_segment)
        .map(|(segment, seed, solve)| {
            segment_solve_stats(
                *segment,
                &clip_segments[*segment],
                false,
                seed.map(|index| &seed_candidates[index]),
                solve.as_ref(),
                focal,
            )
        })
        .collect();
    drop(secondary);
    let mut mark_selected = |from: usize, to: usize| {
        if let Some(seed) = seed_evidence
            .iter_mut()
            .find(|seed| seed.from_frame == from && seed.to_frame == to)
        {
            seed.selected = true;
        }
    };
    for solve in &segment_solves {
        control.check()?;
        if let Some(seed) = &solve.seed_pair {
            mark_selected(seed.from_frame, seed.to_frame);
        }
    }
    if let Some(index) = primary_index {
        let [from, to] = seed_candidates[index].0;
        mark_selected(from, to);
    }
    let best_two_view = primary_index.map(|index| seed_candidates.swap_remove(index));
    drop(seed_candidates);
    let primary_solve = solve_segment(
        best_two_view.as_ref(),
        &keyframe_selection,
        &adjacent_matches,
        &features,
        &revisit_context,
        width,
        height,
        focal as f64,
        false,
        control,
    )?;
    if let Some(segment) = primary_segment.filter(|_| clip_segments.len() > 1) {
        segment_solves.push(segment_solve_stats(
            segment,
            &clip_segments[segment],
            true,
            best_two_view.as_ref(),
            Some(&primary_solve),
            focal,
        ));
        segment_solves.sort_by_key(|solve| solve.segment);
    }
    let (primary_first, primary_last) = primary_segment.filter(|_| clip_segments.len() > 1).map_or(
        (0, request.frames.len() - 1),
        |segment| {
            (
                clip_segments[segment].first_frame,
                clip_segments[segment].last_frame,
            )
        },
    );
    let track_scope = TrackScope {
        spans: primary_solve.multi_view_analysis.track_spans(),
        linked: adjacent_matches
            .iter()
            .map(|matches| !matches.is_empty())
            .collect(),
        primary: (primary_first, primary_last),
    };
    let SegmentSolve {
        mut multi_view_analysis,
        revisits,
        registered_views,
        registered_cameras,
        registered_geometry,
        new_landmarks,
        optimized_seed_points,
        optimized_new_landmark_positions,
        expanded_views,
    } = primary_solve;
    multi_view_analysis.stats.segment_solves = segment_solves;

    let dense_sparse_points: Vec<nalgebra::Vector3<f64>> = optimized_seed_points
        .iter()
        .chain(&optimized_new_landmark_positions)
        .copied()
        .collect();
    let mut multi_view = multi_view_analysis.stats;

    let calibrated_pair = best_two_view.map(|(seed_frames, estimate)| {
        let source_features = &features[seed_frames[0]];
        let frame = &request.frames[seed_frames[0]];
        cameras = seed_pair_cameras(seed_frames, &estimate);
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

        calibrated_pair_stats(seed_frames, &estimate, focal)
    });

    if let Some(pair) = calibrated_pair
        .as_ref()
        .filter(|pair| pair.to_frame > pair.from_frame + 1)
    {
        let rotation_residual = seed_evidence
            .iter()
            .find(|seed| seed.from_frame == pair.from_frame && seed.to_frame == pair.to_frame)
            .and_then(|seed| seed.rotation_only_residual_pixels);
        let segment = frame_segments[pair.from_frame];
        let span = &multi_view.keyframe_selection.segments[segment];
        let promoted = promoted_keyframes.get(&segment).copied().unwrap_or(0);
        warnings.push(format!(
            "No adjacent frame pair of frames {}–{} passed every seed gate, so their calibrated seed pair spans frames {}–{} ({} sampling intervals): matches composed along the tracked links and refined to sub-pixel positions leave {} px after the best pure rotation (requires more than {:.2} px), and the pair passed every two-view gate with a {:.2}° median triangulation angle. {promoted} further frame(s) of that span became registration keyframes.",
            span.first_frame + 1,
            span.last_frame + 1,
            pair.from_frame + 1,
            pair.to_frame + 1,
            pair.to_frame - pair.from_frame,
            rotation_residual.map_or_else(|| "n/a".to_owned(), |value| format!("{value:.2}")),
            two_view::MIN_SEED_ROTATION_RESIDUAL_PIXELS,
            pair.median_triangulation_angle_degrees,
        ));
    }

    if motion_guided_pairs > 0 {
        warnings.push(format!(
        "Motion-guided adjacent matching recovered {motion_guided_pairs} sampled frame pair(s) after the ordinary origin-centered local search was starved. A strict global descriptor consensus only predicts the dominant displacement used to center the existing bounded local search; all calibrated geometry still passes the normal epipolar, PnP, bundle-adjustment, dense-depth, and mesh gates."
    ));
    }

    let segments = &multi_view.keyframe_selection.segments;
    if segments.len() > 1 {
        let spans = segments
            .iter()
            .map(|segment| format!("{}–{}", segment.first_frame + 1, segment.last_frame + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let solved = match multi_view
            .segment_solves
            .iter()
            .find(|solve| solve.primary)
        {
            Some(primary) => {
                let seeded = multi_view
                    .segment_solves
                    .iter()
                    .filter(|solve| solve.seed_pair.is_some())
                    .count();
                let unseeded = segments.len() - seeded;
                let unseeded_note = if unseeded > 0 {
                    format!(" {unseeded} segment(s) had no calibrated seed pair and are reported, not solved.")
                } else {
                    String::new()
                };
                format!(
                    "{seeded} segment(s) with a calibrated seed pair are each solved on their own, in their own arbitrary frame; no segment is forced into another's solve. The displayed geometry, dense depth and mesh come from segment {} (frames {}–{}), which has the strongest seed; per-segment seeds and registration counts are listed in the evidence.{unseeded_note}",
                    primary.segment + 1,
                    primary.first_frame + 1,
                    primary.last_frame + 1,
                )
            }
            None => "No segment yielded a calibrated seed pair, so none is solved; the uncalibrated preview spans all segments.".to_owned(),
        };
        warnings.push(format!(
            "The clip splits into {} segments at hard cuts or discontinuities (frames {spans}). {solved}",
            segments.len()
        ));
    }

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

    // The dense and mesh stages report here, between the camera stage's warnings above
    // and below (see `reconstruction_stages::ordered_warnings`).
    let warnings_before_dense = std::mem::take(&mut warnings);

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

    // The seed's segment is the only one its solve may register into.
    let seed_covers_segment = calibrated_pair.as_ref().is_some_and(|pair| {
        multi_view
            .keyframe_selection
            .segments
            .iter()
            .find(|segment| {
                segment.first_frame <= pair.from_frame && pair.to_frame <= segment.last_frame
            })
            .map_or(pairs.len() <= 1, |segment| {
                segment.last_frame - segment.first_frame <= 1
            })
    });
    let bootstrap = diagnose_bootstrap(
        &seed_evidence,
        calibrated_pair.is_some(),
        seed_covers_segment,
        registered_views.len(),
        multi_view
            .registration_candidates
            .iter()
            .filter(|candidate| candidate.pnp_ready)
            .count(),
    );
    if calibrated_pair.is_some() {
        let pnp_ready = multi_view
            .registration_candidates
            .iter()
            .filter(|candidate| candidate.pnp_ready)
            .count();
        if registered_views.is_empty() {
            let (from, to) = calibrated_pair
                .as_ref()
                .map_or((0, 0), |pair| (pair.from_frame + 1, pair.to_frame + 1));
            warnings.push(format!(
                "Slice 3 finds {pnp_ready} other selected keyframes with enough tracked seed landmarks for a robust PnP attempt, but neither the initial track-based registration nor bounded direct-revisit recovery produced an additional camera pose that passed the deterministic inlier and reprojection gates. The displayed geometry remains the calibrated seed pair, frames {from}–{to}; translation scale is arbitrary, and bundle adjustment requires at least one accepted additional view."
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
        warnings.push(format!(
            "{} This result retains the conservative uncalibrated MVP preview.",
            bootstrap.summary
        ));
    }

    let final_cameras = std::mem::take(&mut multi_view.bundle_adjustment.final_cameras);
    let mut revisits = revisits;
    revisits.release_solver_evidence();
    Ok(SparseReconstruction {
        options,
        image_width: width,
        image_height: height,
        focal_length_pixels: focal,
        cameras,
        points,
        pairs,
        calibrated_pair,
        multi_view,
        revisits,
        registered_views,
        seed_candidates: seed_evidence,
        bootstrap,
        final_cameras: final_cameras.iter().map(StageCamera::from).collect(),
        dense_cameras: registered_geometry.iter().map(StageCamera::from).collect(),
        dense_sparse_points: dense_sparse_points
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect(),
        track_scope,
        expanded_views,
        warnings_before_dense,
        warnings_after_mesh: warnings,
    })
}

fn validate_request(request: &ReconstructionRequest) -> Result<(), String> {
    if request.frames.len() < 2 {
        return Err("at least two sampled frames are required".into());
    }
    // Non-finite options would also make the camera stage artifact unreloadable: JSON
    // has no NaN or infinity, so they would be stored as `null`.
    if !request.options.max_descriptor_distance.is_finite() {
        return Err("max descriptor distance must be finite".into());
    }
    if request.options.max_descriptor_distance <= 0.0 {
        return Err("max descriptor distance must be positive".into());
    }
    if !request.options.ratio_threshold.is_finite() {
        return Err("descriptor ratio threshold must be finite".into());
    }
    if request.options.max_dense_working_set_bytes == 0 {
        return Err("dense working-set budget must be positive".into());
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
