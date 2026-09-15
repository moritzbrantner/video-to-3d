use serde::{
    de::{Deserializer, SeqAccess, Visitor},
    Deserialize, Serialize,
};
use std::fmt;
use video_to_3d_core::feature_analysis::{
    analyze_rgba_pair, FeatureAlgorithm, FeatureAnalysis, FeatureOptions,
};
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct FeatureAnalysisRequest {
    width: u32,
    height: u32,
    #[serde(deserialize_with = "deserialize_rgba_bytes")]
    source_rgba: Vec<u8>,
    #[serde(deserialize_with = "deserialize_rgba_bytes")]
    target_rgba: Vec<u8>,
    #[serde(default)]
    algorithm: FeatureAlgorithm,
    #[serde(default)]
    options: FeatureOptions,
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

    deserializer.deserialize_byte_buf(ByteVisitor)
}

fn browser_serializer() -> serde_wasm_bindgen::Serializer {
    serde_wasm_bindgen::Serializer::new()
        .serialize_missing_as_null(true)
        .serialize_maps_as_objects(true)
}

#[wasm_bindgen]
pub fn analyze_feature_pair(value: JsValue) -> Result<JsValue, JsValue> {
    let request: FeatureAnalysisRequest =
        serde_wasm_bindgen::from_value(value).map_err(|error| {
            JsValue::from_str(&format!("invalid feature-analysis request: {error}"))
        })?;
    let analysis: FeatureAnalysis = analyze_rgba_pair(
        &request.source_rgba,
        &request.target_rgba,
        request.width,
        request.height,
        request.algorithm,
        request.options,
    )
    .map_err(|error| JsValue::from_str(&error))?;

    analysis.serialize(&browser_serializer()).map_err(|error| {
        JsValue::from_str(&format!("failed to serialize feature analysis: {error}"))
    })
}
