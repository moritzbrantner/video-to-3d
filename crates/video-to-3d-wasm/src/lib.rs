use video_to_3d_core::{reconstruct, ReconstructionRequest, ReconstructionResult};
use wasm_bindgen::prelude::*;

const PAN_RECOVERY_WIDTH_FRACTION: f32 = 0.30;
const PAN_RECOVERY_MAX_RADIUS: u32 = 112;
const PAN_RECOVERY_RATIO_THRESHOLD: f32 = 0.78;
const REGISTRATION_RECOVERY_RATIO_THRESHOLD: f32 = 0.74;
const REGISTRATION_RECOVERY_MAX_FEATURES: usize = 640;
const REGISTRATION_RECOVERY_MIN_FEATURE_DISTANCE: u32 = 5;
const MIN_TRACK_MATCHES: usize = 8;
const MIN_TRACK_OVERLAP: f32 = 0.18;

#[wasm_bindgen]
pub fn reconstruct_sequence(value: JsValue) -> Result<JsValue, JsValue> {
    let request: ReconstructionRequest = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("invalid reconstruction request: {error}")))?;
    let result = reconstruct_with_registration_recovery(&request)
        .map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&result)
        .map_err(|error| JsValue::from_str(&format!("failed to serialize reconstruction: {error}")))
}

fn reconstruct_with_registration_recovery(
    request: &ReconstructionRequest,
) -> Result<ReconstructionResult, String> {
    let initial = reconstruct(request)?;
    if !needs_registration_recovery(request, &initial) {
        return Ok(initial);
    }

    let Some(first_frame) = request.frames.first() else {
        return Ok(initial);
    };
    let retry_radius =
        pan_recovery_radius(first_frame.width, request.options.match_radius, &initial);
    let mut best = initial;
    let mut selected_recovery = None;

    if retry_radius > request.options.match_radius {
        let mut pan_request = request.clone();
        pan_request.options.match_radius = retry_radius;
        pan_request.options.ratio_threshold = pan_request
            .options
            .ratio_threshold
            .min(PAN_RECOVERY_RATIO_THRESHOLD);
        let pan_retry = reconstruct(&pan_request)?;
        if reconstruction_score(&pan_retry) > reconstruction_score(&best) {
            best = pan_retry;
            selected_recovery = Some("bounded displacement-informed match-radius recovery");
        }
    }

    if needs_registration_recovery(request, &best) {
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

        let registration_retry = reconstruct(&registration_request)?;
        if reconstruction_score(&registration_retry) > reconstruction_score(&best) {
            best = registration_retry;
            selected_recovery = Some(
                "bounded registration recovery with denser features and stricter descriptor ambiguity filtering",
            );
        }
    }

    if let Some(recovery) = selected_recovery {
        best.warnings.push(format!(
            "The browser adapter selected {recovery} after the ordinary reconstruction lacked sufficient accepted multi-view evidence. All seed-pair, PnP, bundle-adjustment, dense-depth, and mesh acceptance gates remain owned by video-to-3d-core."
        ));
    }

    Ok(best)
}

fn needs_registration_recovery(
    request: &ReconstructionRequest,
    result: &ReconstructionResult,
) -> bool {
    if request.frames.len() < 3 {
        return false;
    }

    let starved_adjacent_pair = result
        .pairs
        .iter()
        .any(|pair| pair.matches < MIN_TRACK_MATCHES || pair.overlap_ratio < MIN_TRACK_OVERLAP);
    let missing_seed = result.calibrated_pair.is_none();
    let missing_selected_view = result.calibrated_pair.is_some()
        && result.multi_view.registration_candidates.len() > result.registered_views.len();
    let tracks_end_early = request.frames.len() >= 4 && result.multi_view.longest_track < 4;

    missing_seed || starved_adjacent_pair || missing_selected_view || tracks_end_early
}

fn pan_recovery_radius(width: u32, current_radius: u32, result: &ReconstructionResult) -> u32 {
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

fn accepted_registered_camera_count(result: &ReconstructionResult) -> usize {
    if result.calibrated_pair.is_some() {
        result.cameras.len()
    } else {
        0
    }
}

fn reconstruction_score(result: &ReconstructionResult) -> [usize; 10] {
    [
        usize::from(result.calibrated_pair.is_some()),
        accepted_registered_camera_count(result),
        result.registered_views.len(),
        usize::from(result.multi_view.bundle_adjustment.accepted),
        result.multi_view.tracks_three_plus,
        result.multi_view.longest_track,
        result.multi_view.linked_pairs,
        result.points.len(),
        result.dense_points.len(),
        result.mesh_triangles.len(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_result_with_motion(median_motion: f32) -> ReconstructionResult {
        let request = ReconstructionRequest {
            frames: vec![
                video_to_3d_core::FrameInput {
                    width: 96,
                    height: 80,
                    rgba: vec![0; 96 * 80 * 4],
                },
                video_to_3d_core::FrameInput {
                    width: 96,
                    height: 80,
                    rgba: vec![0; 96 * 80 * 4],
                },
            ],
            options: Default::default(),
        };
        let mut result =
            reconstruct(&request).expect("blank fixture should reconstruct deterministically");
        if let Some(pair) = result.pairs.first_mut() {
            pair.median_motion = median_motion;
        }
        result
    }

    #[test]
    fn pan_recovery_radius_scales_with_analysis_width() {
        let result = empty_result_with_motion(0.0);
        assert_eq!(pan_recovery_radius(360, 42, &result), 108);
        assert_eq!(pan_recovery_radius(180, 42, &result), 54);
        assert_eq!(pan_recovery_radius(96, 42, &result), 42);
    }

    #[test]
    fn pan_recovery_radius_accounts_for_observed_displacement() {
        let result = empty_result_with_motion(44.0);
        assert_eq!(pan_recovery_radius(240, 42, &result), 104);
    }

    #[test]
    fn pan_recovery_radius_never_shrinks_existing_search() {
        let result = empty_result_with_motion(0.0);
        assert_eq!(pan_recovery_radius(360, 128, &result), 128);
    }
}
