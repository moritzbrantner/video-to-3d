use serde::Serialize;
use video_to_3d_core::{reconstruct_browser, ReconstructionRequest};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn reconstruct_sequence(value: JsValue) -> Result<JsValue, JsValue> {
    let request: ReconstructionRequest = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("invalid reconstruction request: {error}")))?;
    let result = reconstruct_browser(&request).map_err(|error| JsValue::from_str(&error))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    result
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("failed to serialize reconstruction: {error}")))
}
