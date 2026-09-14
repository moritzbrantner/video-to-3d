use serde::{
    de::{Deserializer, SeqAccess, Visitor},
    Deserialize, Serialize,
};
use std::fmt;
use video_to_3d_core::{
    reconstruct_browser, FrameInput, ReconstructionOptions, ReconstructionRequest,
};
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct WasmFrameInput {
    width: u32,
    height: u32,
    #[serde(deserialize_with = "deserialize_rgba_bytes")]
    rgba: Vec<u8>,
}

#[derive(Deserialize)]
struct WasmReconstructionRequest {
    frames: Vec<WasmFrameInput>,
    #[serde(default)]
    options: ReconstructionOptions,
}

fn deserialize_rgba_bytes<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    struct ByteVisitor;

    impl<'de> Visitor<'de> for ByteVisitor {
        type Value = Vec<u8>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an RGBA byte buffer or byte sequence")
        }

        fn visit_byte_buf<E>(self, bytes: Vec<u8>) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(bytes)
        }

        fn visit_bytes<E>(self, bytes: &[u8]) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(bytes.to_vec())
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut bytes = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
            while let Some(byte) = sequence.next_element::<u8>()? {
                bytes.push(byte);
            }
            Ok(bytes)
        }
    }

    // serde-wasm-bindgen maps Uint8Array/ArrayBuffer to visit_byte_buf, which
    // performs one bulk copy instead of iterating through every JavaScript value.
    // visit_seq preserves the legacy number[] request shape for compatibility and
    // for the benchmark comparison.
    deserializer.deserialize_byte_buf(ByteVisitor)
}

impl From<WasmReconstructionRequest> for ReconstructionRequest {
    fn from(request: WasmReconstructionRequest) -> Self {
        Self {
            frames: request
                .frames
                .into_iter()
                .map(|frame| FrameInput {
                    width: frame.width,
                    height: frame.height,
                    rgba: frame.rgba,
                })
                .collect(),
            options: request.options,
        }
    }
}

#[wasm_bindgen]
pub fn reconstruct_sequence(value: JsValue) -> Result<JsValue, JsValue> {
    let request: WasmReconstructionRequest = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("invalid reconstruction request: {error}")))?;
    let request = ReconstructionRequest::from(request);
    let result = reconstruct_browser(&request).map_err(|error| JsValue::from_str(&error))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    result
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("failed to serialize reconstruction: {error}")))
}
