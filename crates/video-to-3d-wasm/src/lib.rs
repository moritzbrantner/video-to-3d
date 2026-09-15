use serde::{
    de::{Deserializer, SeqAccess, Visitor},
    Deserialize, Serialize,
};
use std::fmt;
use video_to_3d_core::{
    reconstruct_browser, CalibratedPairStats, CameraPipelineState, CameraPose, DenseStats,
    FrameInput, MeshStats, MeshTriangle, MultiViewStats, PairStats, Point3, ReconstructionOptions,
    ReconstructionRequest, RegisteredViewStats, RevisitStats,
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

struct ByteBuffer(Vec<u8>);

impl Serialize for ByteBuffer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

#[derive(Serialize)]
struct PackedDensePointBuffer {
    values_f32_le: ByteBuffer,
    rgb: ByteBuffer,
    length: usize,
}

#[derive(Serialize)]
struct PackedMeshTriangleBuffer {
    indices_u32_le: ByteBuffer,
    confidence_f32_le: ByteBuffer,
    length: usize,
}

#[derive(Serialize)]
struct WasmBrowserReconstructionResult<'a> {
    cameras: &'a [CameraPose],
    points: &'a [Point3],
    dense_points: PackedDensePointBuffer,
    dense: &'a DenseStats,
    mesh_triangles: PackedMeshTriangleBuffer,
    mesh: &'a MeshStats,
    pairs: &'a [PairStats],
    calibrated_pair: &'a Option<CalibratedPairStats>,
    multi_view: &'a MultiViewStats,
    revisits: &'a RevisitStats,
    registered_views: &'a [RegisteredViewStats],
    warnings: &'a [String],
    camera_state: &'a CameraPipelineState,
}

#[derive(Serialize)]
struct GeometryBenchmarkObjectPayload {
    dense_points: Vec<Point3>,
    mesh_triangles: Vec<MeshTriangle>,
}

#[derive(Serialize)]
struct GeometryBenchmarkPackedPayload {
    dense_points: PackedDensePointBuffer,
    mesh_triangles: PackedMeshTriangleBuffer,
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

fn browser_serializer() -> serde_wasm_bindgen::Serializer {
    // Preserve the established plain-object / null shape, but unlike the
    // json_compatible preset keep explicit byte buffers as Uint8Array values.
    serde_wasm_bindgen::Serializer::new()
        .serialize_missing_as_null(true)
        .serialize_maps_as_objects(true)
}

fn pack_dense_points(points: &[Point3]) -> PackedDensePointBuffer {
    let mut values_f32_le = Vec::with_capacity(points.len() * 4 * size_of::<f32>());
    let mut rgb = Vec::with_capacity(points.len() * 3);
    for point in points {
        values_f32_le.extend_from_slice(&point.x.to_le_bytes());
        values_f32_le.extend_from_slice(&point.y.to_le_bytes());
        values_f32_le.extend_from_slice(&point.z.to_le_bytes());
        values_f32_le.extend_from_slice(&point.confidence.to_le_bytes());
        rgb.extend_from_slice(&[point.r, point.g, point.b]);
    }
    PackedDensePointBuffer {
        values_f32_le: ByteBuffer(values_f32_le),
        rgb: ByteBuffer(rgb),
        length: points.len(),
    }
}

fn pack_mesh_triangles(
    triangles: &[MeshTriangle],
    point_count: usize,
) -> Result<PackedMeshTriangleBuffer, JsValue> {
    let mut indices_u32_le = Vec::with_capacity(triangles.len() * 3 * size_of::<u32>());
    let mut confidence_f32_le = Vec::with_capacity(triangles.len() * size_of::<f32>());
    for triangle in triangles {
        for index in [triangle.a, triangle.b, triangle.c] {
            if index >= point_count {
                return Err(JsValue::from_str(
                    "failed to serialize reconstruction: mesh triangle references a missing dense point",
                ));
            }
            let index = u32::try_from(index).map_err(|_| {
                JsValue::from_str(
                    "failed to serialize reconstruction: dense point index exceeds browser transport range",
                )
            })?;
            indices_u32_le.extend_from_slice(&index.to_le_bytes());
        }
        confidence_f32_le.extend_from_slice(&triangle.confidence.to_le_bytes());
    }
    Ok(PackedMeshTriangleBuffer {
        indices_u32_le: ByteBuffer(indices_u32_le),
        confidence_f32_le: ByteBuffer(confidence_f32_le),
        length: triangles.len(),
    })
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
    payload.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("failed to serialize geometry benchmark: {error}"))
    })
}

#[wasm_bindgen]
pub fn benchmark_packed_geometry_result(
    point_count: u32,
    triangle_count: u32,
) -> Result<JsValue, JsValue> {
    let point_count = validate_geometry_benchmark_counts(point_count)?;
    let triangle_count = triangle_count as usize;
    let points: Vec<_> = (0..point_count).map(benchmark_point).collect();
    let triangles: Vec<_> = (0..triangle_count)
        .map(|index| benchmark_triangle(index, point_count))
        .collect();
    let payload = GeometryBenchmarkPackedPayload {
        dense_points: pack_dense_points(&points),
        mesh_triangles: pack_mesh_triangles(&triangles, points.len())?,
    };
    let serializer = browser_serializer();
    payload.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "failed to serialize packed geometry benchmark: {error}"
        ))
    })
}

#[wasm_bindgen]
pub fn reconstruct_sequence(value: JsValue) -> Result<JsValue, JsValue> {
    let request: WasmReconstructionRequest = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("invalid reconstruction request: {error}")))?;
    let request = ReconstructionRequest::from(request);
    let result = reconstruct_browser(&request).map_err(|error| JsValue::from_str(&error))?;
    let reconstruction = &result.reconstruction;
    let payload = WasmBrowserReconstructionResult {
        cameras: &reconstruction.cameras,
        points: &reconstruction.points,
        dense_points: pack_dense_points(&reconstruction.dense_points),
        dense: &reconstruction.dense,
        mesh_triangles: pack_mesh_triangles(
            &reconstruction.mesh_triangles,
            reconstruction.dense_points.len(),
        )?,
        mesh: &reconstruction.mesh,
        pairs: &reconstruction.pairs,
        calibrated_pair: &reconstruction.calibrated_pair,
        multi_view: &reconstruction.multi_view,
        revisits: &reconstruction.revisits,
        registered_views: &reconstruction.registered_views,
        warnings: &reconstruction.warnings,
        camera_state: &result.camera_state,
    };
    let serializer = browser_serializer();
    payload
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("failed to serialize reconstruction: {error}")))
}
