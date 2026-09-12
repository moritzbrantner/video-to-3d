use video_to_3d_core::{reconstruct, ReconstructionRequest, ReconstructionResult};
use wasm_bindgen::prelude::*;

const PAN_RECOVERY_WIDTH_FRACTION: f32 = 0.30;
const PAN_RECOVERY_MAX_RADIUS: u32 = 112;
const PAN_RECOVERY_RATIO_THRESHOLD: f32 = 0.78;
const MIN_TRACK_MATCHES: usize = 8;
const MIN_TRACK_OVERLAP: f32 = 0.18;

#[wasm_bindgen]
pub fn reconstruct_sequence(value: JsValue) -> Result<JsValue, JsValue> {
    let request: ReconstructionRequest = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("invalid reconstruction request: {error}")))?;
    let result =
        reconstruct_with_pan_recovery(&request).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&result)
        .map_err(|error| JsValue::from_str(&format!("failed to serialize reconstruction: {error}")))
}

fn reconstruct_with_pan_recovery(
    request: &ReconstructionRequest,
) -> Result<ReconstructionResult, String> {
    let initial = reconstruct(request)?;
    if !needs_pan_recovery(request, &initial) {
        return Ok(initial);
    }

    let Some(first_frame) = request.frames.first() else {
        return Ok(initial);
    };
    let retry_radius = pan_recovery_radius(first_frame.width, request.options.match_radius);
    if retry_radius <= request.options.match_radius {
        return Ok(initial);
    }

    let mut retry_request = request.clone();
    retry_request.options.match_radius = retry_radius;
    retry_request.options.ratio_threshold = retry_request
        .options
        .ratio_threshold
        .min(PAN_RECOVERY_RATIO_THRESHOLD);
    let retry = reconstruct(&retry_request)?;

    if reconstruction_score(&retry) > reconstruction_score(&initial) {
        Ok(retry)
    } else {
        Ok(initial)
    }
}

fn needs_pan_recovery(request: &ReconstructionRequest, result: &ReconstructionResult) -> bool {
    if request.frames.len() < 3 {
        return false;
    }

    let starved_adjacent_pair = result
        .pairs
        .iter()
        .any(|pair| pair.matches < MIN_TRACK_MATCHES || pair.overlap_ratio < MIN_TRACK_OVERLAP);
    let missing_selected_view = result.calibrated_pair.is_some()
        && result.multi_view.registration_candidates.len() > result.registered_views.len();
    let tracks_end_early = request.frames.len() >= 4 && result.multi_view.longest_track < 4;

    starved_adjacent_pair || missing_selected_view || tracks_end_early
}

fn pan_recovery_radius(width: u32, current_radius: u32) -> u32 {
    if current_radius >= PAN_RECOVERY_MAX_RADIUS {
        return current_radius;
    }

    let scaled = (width as f32 * PAN_RECOVERY_WIDTH_FRACTION).round() as u32;
    scaled.clamp(current_radius, PAN_RECOVERY_MAX_RADIUS)
}

fn reconstruction_score(result: &ReconstructionResult) -> [usize; 9] {
    [
        usize::from(result.calibrated_pair.is_some()),
        result.registered_views.len(),
        result.cameras.len(),
        usize::from(result.multi_view.bundle_adjustment.accepted),
        result.multi_view.tracks_three_plus,
        result.multi_view.longest_track,
        result.multi_view.linked_pairs,
        result.points.len(),
        result.dense_points.len(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pan_recovery_radius_scales_with_analysis_width() {
        assert_eq!(pan_recovery_radius(360, 42), 108);
        assert_eq!(pan_recovery_radius(180, 42), 54);
        assert_eq!(pan_recovery_radius(96, 42), 42);
    }

    #[test]
    fn pan_recovery_radius_never_shrinks_existing_search() {
        assert_eq!(pan_recovery_radius(360, 128), 128);
    }
}
