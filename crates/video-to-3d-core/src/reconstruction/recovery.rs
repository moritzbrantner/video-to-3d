//! The camera stage entry points: one ordinary solve, then the bounded recovery retries
//! when it lacks multi-view evidence.

use super::control::ReconstructionControl;
use super::solve::{reconstruct_once_with_control, MIN_TRACK_MATCHES, MIN_TRACK_OVERLAP};
use super::types::{ReconstructionRequest, ReconstructionResult, SegmentSolveStats};
use crate::reconstruction_stages;
use crate::reconstruction_stages::SparseReconstruction;

const PAN_RECOVERY_WIDTH_FRACTION: f32 = 0.30;
const PAN_RECOVERY_MAX_RADIUS: u32 = 112;
const PAN_RECOVERY_RATIO_THRESHOLD: f32 = 0.78;
const REGISTRATION_RECOVERY_RATIO_THRESHOLD: f32 = 0.74;
const REGISTRATION_RECOVERY_MAX_FEATURES: usize = 640;
const REGISTRATION_RECOVERY_MIN_FEATURE_DISTANCE: u32 = 5;

/// The classic pipeline in one call. It composes the same three stage cores as the public
/// staged entry points ([`reconstruct_cameras`](crate::reconstruct_cameras) → [`reconstruct_dense`](crate::reconstruct_dense) →
/// [`reconstruct_mesh`](crate::reconstruct_mesh), joined by [`compose_reconstruction`](crate::compose_reconstruction)): registration with its
/// bounded recovery, dense evidence of the selected solve, then the mesh and surface
/// consolidation. It only skips the stage artifacts' identity hashing and storage.
pub fn reconstruct(request: &ReconstructionRequest) -> Result<ReconstructionResult, String> {
    let sparse = reconstruct_sparse(request)?;
    let dense = reconstruction_stages::estimate_dense_evidence(&request.frames, &sparse);
    reconstruction_stages::finish_reconstruction(sparse, dense)
}

/// The camera stage: one ordinary solve plus, when it lacks multi-view evidence, the
/// bounded recovery retries. Recovery ranks sparse evidence only, so no candidate needs
/// dense depth or a mesh; the selected solve records its effective options.
pub(crate) fn reconstruct_sparse(
    request: &ReconstructionRequest,
) -> Result<SparseReconstruction, String> {
    reconstruct_sparse_with_control(request, &mut ReconstructionControl::new(&mut || false))
}

pub(crate) fn reconstruct_sparse_with_control(
    request: &ReconstructionRequest,
    control: &mut ReconstructionControl<'_>,
) -> Result<SparseReconstruction, String> {
    control.check()?;
    let initial = reconstruct_once_with_control(request, control)?;
    if !needs_registration_recovery(request, &initial) {
        return Ok(initial);
    }

    let Some(first_frame) = request.frames.first() else {
        return Ok(initial);
    };
    let retry_radius =
        pan_recovery_radius(first_frame.width, request.options.match_radius, &initial);
    // Every candidate's track evidence is compared over the first solve's primary frames,
    // even when a retry segments the clip differently.
    let scope = initial.track_scope.primary;
    let score = |result: &SparseReconstruction| reconstruction_score(result, scope);
    let mut best = initial;
    let mut selected_recovery = None;

    if retry_radius > request.options.match_radius {
        let mut pan_request = request.clone();
        pan_request.options.match_radius = retry_radius;
        pan_request.options.ratio_threshold = pan_request
            .options
            .ratio_threshold
            .min(PAN_RECOVERY_RATIO_THRESHOLD);
        let pan_retry = reconstruct_once_with_control(&pan_request, control)?;
        if score(&pan_retry) > score(&best) {
            best = pan_retry;
            selected_recovery = Some("bounded displacement-informed match-radius recovery");
        }
    }

    let registration_request = needs_registration_recovery(request, &best).then(|| {
        let mut registration_request = request.clone();
        registration_request.options.match_radius = retry_radius;
        registration_request.options.max_features = registration_request
            .options
            .max_features
            .max(REGISTRATION_RECOVERY_MAX_FEATURES);
        registration_request.options.min_feature_distance = registration_request
            .options
            .min_feature_distance
            .min(REGISTRATION_RECOVERY_MIN_FEATURE_DISTANCE);
        registration_request.options.ratio_threshold = registration_request
            .options
            .ratio_threshold
            .min(REGISTRATION_RECOVERY_RATIO_THRESHOLD);
        registration_request
    });

    // A request already at the recovery settings would rerun the same deterministic solve.
    if let Some(registration_request) = registration_request.filter(|retry| {
        let (a, b) = (&retry.options, &request.options);
        (a.match_radius, a.max_features, a.min_feature_distance)
            != (b.match_radius, b.max_features, b.min_feature_distance)
            || a.ratio_threshold != b.ratio_threshold
    }) {
        let registration_retry = reconstruct_once_with_control(&registration_request, control)?;
        if score(&registration_retry) > score(&best) {
            best = registration_retry;
            selected_recovery = Some(
                "bounded registration recovery with denser features and stricter descriptor ambiguity filtering",
            );
        }
    }

    if let Some(recovery) = selected_recovery {
        best.warnings_after_mesh.push(format!(
            "The Rust core selected {recovery} after the ordinary reconstruction lacked sufficient accepted multi-view evidence. Seed-pair, PnP, bundle-adjustment, dense-depth, and mesh acceptance gates remain unchanged."
        ));
    }

    Ok(best)
}

pub(super) fn needs_registration_recovery(
    request: &ReconstructionRequest,
    result: &SparseReconstruction,
) -> bool {
    if request.frames.len() < 3 {
        return false;
    }
    let starved_adjacent_pair = result
        .pairs
        .iter()
        .any(|pair| pair.matches < MIN_TRACK_MATCHES || pair.overlap_ratio < MIN_TRACK_OVERLAP);
    let missing_seed = result.calibrated_pair.is_none();
    // Views that only incremental expansion reached still deserve the denser recovery
    // solve; scoring keeps whichever candidate is stronger.
    let missing_selected_view = result.calibrated_pair.is_some()
        && (result.multi_view.registration_candidates.len() > result.registered_views.len()
            || result.expanded_views > 0);
    let tracks_end_early = request.frames.len() >= 4 && result.multi_view.longest_track < 4;
    // A split clip solves its other segments on their own, so a retry can also help them.
    // A one-frame segment has no pair to seed. Views recovered from revisit evidence need
    // not be PnP-ready candidates, so they cannot cover a failed one.
    let secondary_segment_short = secondary_segment_solves(result).any(|solve| {
        (solve.seed_pair.is_none() && solve.last_frame > solve.first_frame)
            || solve.pnp_ready_candidates
                > solve
                    .registered_views
                    .saturating_sub(solve.recovered_from_revisit)
    });
    missing_seed
        || starved_adjacent_pair
        || missing_selected_view
        || tracks_end_early
        || secondary_segment_short
}

pub(super) fn secondary_segment_solves(
    result: &SparseReconstruction,
) -> impl Iterator<Item = &SegmentSolveStats> {
    result
        .multi_view
        .segment_solves
        .iter()
        .filter(|solve| !solve.primary)
}

pub(super) fn pan_recovery_radius(
    width: u32,
    current_radius: u32,
    result: &SparseReconstruction,
) -> u32 {
    if current_radius >= PAN_RECOVERY_MAX_RADIUS {
        return current_radius;
    }
    let width_scaled = (width as f32 * PAN_RECOVERY_WIDTH_FRACTION).round() as u32;
    let observed_motion = result
        .pairs
        .iter()
        .map(|pair| pair.median_motion)
        .filter(|motion| motion.is_finite() && *motion > 0.0)
        .fold(0.0f32, f32::max);
    let motion_scaled = (observed_motion * 2.0 + 16.0).ceil() as u32;
    width_scaled
        .max(motion_scaled)
        .clamp(current_radius, PAN_RECOVERY_MAX_RADIUS)
}

fn accepted_registered_camera_count(result: &SparseReconstruction) -> usize {
    if result.calibrated_pair.is_some() {
        result.cameras.len()
    } else {
        0
    }
}

pub(super) fn reconstruction_score(
    result: &SparseReconstruction,
    scope: (usize, usize),
) -> [usize; 10] {
    // Recovery chooses the strongest sparse reconstruction before dense geometry is accepted.
    // Dense output may be suppressed by the working-set budget, so it cannot safely decide which
    // camera/point solution survives. Every field up to the sparse points describes the primary
    // solve, with track statistics counted only within `scope`, the frames every compared
    // candidate shares; the other segments of a split clip only break ties, so a retry never
    // trades primary quality for secondary gains.
    let secondary = |measure: fn(&SegmentSolveStats) -> usize| {
        secondary_segment_solves(result).map(measure).sum::<usize>()
    };
    let [tracks_three_plus, longest_track, linked_pairs] = result.track_scope.stats(scope);
    [
        usize::from(result.calibrated_pair.is_some()),
        accepted_registered_camera_count(result),
        result.registered_views.len(),
        usize::from(result.multi_view.bundle_adjustment.accepted),
        tracks_three_plus,
        longest_track,
        linked_pairs,
        result.points.len(),
        // Cameras rather than seeded-segment counts: a retry may segment the clip differently,
        // and fragmenting one range into more seeded segments must not win by itself.
        secondary(|solve| solve.cameras.len()),
        secondary(|solve| solve.sparse_points),
    ]
}
