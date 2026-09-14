use js_sys::{Float32Array, Object, Reflect, Uint32Array, Uint8Array};
use serde::{
    de::{Deserializer, SeqAccess, Visitor},
    Deserialize, Serialize,
};
use std::fmt;
use video_to_3d_core::{
    reconstruct_browser, FrameInput, MeshTriangle, Point3, ReconstructionOptions,
    ReconstructionRequest,
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

#[derive(Serialize)]
struct GeometryBenchmarkObjectPayload {
    dense_points: Vec<Point3>,
    mesh_triangles: Vec<MeshTriangle>,
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

fn benchmark_point(index: usize) -> Point3 {
    Point3 {
        x: index as f32 * 0.001,
        y: (index % 97) as f32 * 0.003,
        z: 1.0 + (index % 31) as f32 * 0.002,
        confidence: 0.75 + (index % 20) as f32 * 0.01,
        r: (index % 251) as u8,
        g: ((index * 3) % 251) as u8,
        b: ((index * 7) % 251) as u8,
    }
}

fn benchmark_triangle(index: usize, point_count: usize) -> MeshTriangle {
    let a = index % point_count;
    MeshTriangle {
        a,
        b: (a + 1) % point_count,
        c: (a + 2) % point_count,
        confidence: 0.8 + (index % 10) as f32 * 0.01,
    }
}

fn validate_geometry_benchmark_counts(point_count: u32) -> Result<usize, JsValue> {
    let point_count = point_count as usize;
    if point_count < 3 {
        return Err(JsValue::from_str(
            "geometry boundary benchmark requires at least three points",
        ));
    }
    Ok(point_count)
}

fn set_object_property(object: &Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    Reflect::set(object, &JsValue::from_str(key), value).map(|_| ())
}

#[wasm_bindgen]
pub fn benchmark_object_geometry_result(
    point_count: u32,
    triangle_count: u32,
) -> Result<JsValue, JsValue> {
    let point_count = validate_geometry_benchmark_counts(point_count)?;
    let triangle_count = triangle_count as usize;
    let payload = GeometryBenchmarkObjectPayload {
        dense_points: (0..point_count).map(benchmark_point).collect(),
        mesh_triangles: (0..triangle_count)
            .map(|index| benchmark_triangle(index, point_count))
            .collect(),
    };
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    payload
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("failed to serialize geometry benchmark: {error}")))
}

#[wasm_bindgen]
pub fn benchmark_packed_geometry_result(
    point_count: u32,
    triangle_count: u32,
) -> Result<JsValue, JsValue> {
    let point_count = validate_geometry_benchmark_counts(point_count)?;
    let triangle_count = triangle_count as usize;

    let mut point_f32 = Vec::with_capacity(point_count * 4);
    let mut point_rgb = Vec::with_capacity(point_count * 3);
    for index in 0..point_count {
        let point = benchmark_point(index);
        point_f32.extend_from_slice(&[point.x, point.y, point.z, point.confidence]);
        point_rgb.extend_from_slice(&[point.r, point.g, point.b]);
    }

    let mut triangle_indices = Vec::with_capacity(triangle_count * 3);
    let mut triangle_confidence = Vec::with_capacity(triangle_count);
    for index in 0..triangle_count {
        let triangle = benchmark_triangle(index, point_count);
        triangle_indices.extend_from_slice(&[
            triangle.a as u32,
            triangle.b as u32,
            triangle.c as u32,
        ]);
        triangle_confidence.push(triangle.confidence);
    }

    let object = Object::new();
    let point_f32: JsValue = Float32Array::from(point_f32.as_slice()).into();
    let point_rgb: JsValue = Uint8Array::from(point_rgb.as_slice()).into();
    let triangle_indices: JsValue = Uint32Array::from(triangle_indices.as_slice()).into();
    let triangle_confidence: JsValue =
        Float32Array::from(triangle_confidence.as_slice()).into();
    set_object_property(&object, "point_f32", &point_f32)?;
    set_object_property(&object, "point_rgb", &point_rgb)?;
    set_object_property(&object, "triangle_indices", &triangle_indices)?;
    set_object_property(&object, "triangle_confidence", &triangle_confidence)?;
    Ok(object.into())
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
