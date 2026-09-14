import assert from "node:assert/strict";
import init, {
  benchmark_object_geometry_result,
  benchmark_packed_geometry_result,
  reconstruct_sequence,
} from "../apps/web/public/wasm/video_to_3d_wasm.js";
import { normalizeWasmReconstruction } from "../apps/web/src/reconstruction";

const width = 360;
const height = 203;
const frameCount = 18;
const bytesPerFrame = width * height * 4;

const options = {
  max_features: 320,
  min_feature_distance: 7,
  descriptor_radius: 3,
  match_radius: 42,
  max_descriptor_distance: 36,
  ratio_threshold: 0.82,
};

const typedFrames = Array.from({ length: frameCount }, (_, frameIndex) => {
  const rgba = new Uint8Array(bytesPerFrame);
  rgba.fill(120 + (frameIndex % 3));
  for (let index = 3; index < rgba.length; index += 4) {
    rgba[index] = 255;
  }
  return { width, height, rgba };
});

const legacyMaterializationStarted = performance.now();
const legacyFrames = typedFrames.map(({ width, height, rgba }) => ({
  width,
  height,
  rgba: Array.from(rgba),
}));
const legacyMaterializationMs = performance.now() - legacyMaterializationStarted;

const wasmPath = new URL(
  "../apps/web/public/wasm/video_to_3d_wasm_bg.wasm",
  import.meta.url,
);
const wasmBytes = await Bun.file(wasmPath).arrayBuffer();
await init({ module_or_path: wasmBytes });

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

function reconstruct(frames: unknown[]): { elapsedMs: number; result: unknown } {
  const started = performance.now();
  const result = normalizeWasmReconstruction(
    reconstruct_sequence({ frames, options }),
  );
  return { elapsedMs: performance.now() - started, result };
}

// Warm the generated WASM glue and reconstruction code on a small fixture before measuring.
reconstruct(legacyFrames.slice(0, 2));
reconstruct(typedFrames.slice(0, 2));

const legacyRuns: number[] = [];
const typedRuns: number[] = [];
let legacyResult: unknown;
let typedResult: unknown;
for (let iteration = 0; iteration < 3; iteration += 1) {
  const first = iteration % 2 === 0 ? "legacy" : "typed";
  for (const representation of [first, first === "legacy" ? "typed" : "legacy"] as const) {
    const measurement = reconstruct(
      representation === "legacy" ? legacyFrames : typedFrames,
    );
    if (representation === "legacy") {
      legacyRuns.push(measurement.elapsedMs);
      legacyResult = measurement.result;
    } else {
      typedRuns.push(measurement.elapsedMs);
      typedResult = measurement.result;
    }
  }
}

assert.deepStrictEqual(
  typedResult,
  legacyResult,
  "typed-byte and number-array requests must produce identical reconstruction results",
);

const legacyMedianMs = median(legacyRuns);
const typedMedianMs = median(typedRuns);
const totalBytes = bytesPerFrame * frameCount;

console.log(
  JSON.stringify(
    {
      benchmark: "wasm-reconstruction-input-boundary",
      width,
      height,
      frame_count: frameCount,
      input_bytes: totalBytes,
      legacy_number_array_materialization_ms: legacyMaterializationMs,
      legacy_number_array_wasm_median_ms: legacyMedianMs,
      typed_array_wasm_median_ms: typedMedianMs,
      typed_array_call_speedup: legacyMedianMs / typedMedianMs,
      note: "Timing is informational only; CI has no wall-clock pass/fail threshold.",
    },
    null,
    2,
  ),
);

type ObjectGeometryPayload = {
  dense_points: Array<{
    x: number;
    y: number;
    z: number;
    confidence: number;
    r: number;
    g: number;
    b: number;
  }>;
  mesh_triangles: Array<{
    a: number;
    b: number;
    c: number;
    confidence: number;
  }>;
};

type PackedGeometryPayload = {
  point_f32: Float32Array;
  point_rgb: Uint8Array;
  triangle_indices: Uint32Array;
  triangle_confidence: Float32Array;
};

function objectGeometry(pointCount: number, triangleCount: number) {
  const started = performance.now();
  const result = benchmark_object_geometry_result(
    pointCount,
    triangleCount,
  ) as ObjectGeometryPayload;
  return { elapsedMs: performance.now() - started, result };
}

function packedGeometry(pointCount: number, triangleCount: number) {
  const started = performance.now();
  const result = benchmark_packed_geometry_result(
    pointCount,
    triangleCount,
  ) as PackedGeometryPayload;
  return { elapsedMs: performance.now() - started, result };
}

function assertEquivalentGeometry(
  object: ObjectGeometryPayload,
  packed: PackedGeometryPayload,
  pointCount: number,
  triangleCount: number,
): void {
  assert.equal(object.dense_points.length, pointCount);
  assert.equal(object.mesh_triangles.length, triangleCount);
  assert.equal(packed.point_f32.length, pointCount * 4);
  assert.equal(packed.point_rgb.length, pointCount * 3);
  assert.equal(packed.triangle_indices.length, triangleCount * 3);
  assert.equal(packed.triangle_confidence.length, triangleCount);

  const point = object.dense_points[0];
  const triangle = object.mesh_triangles[0];
  assert.ok(point && triangle);
  assert.equal(packed.point_f32[0], point.x);
  assert.equal(packed.point_f32[1], point.y);
  assert.equal(packed.point_f32[2], point.z);
  assert.equal(packed.point_f32[3], point.confidence);
  assert.equal(packed.point_rgb[0], point.r);
  assert.equal(packed.point_rgb[1], point.g);
  assert.equal(packed.point_rgb[2], point.b);
  assert.equal(packed.triangle_indices[0], triangle.a);
  assert.equal(packed.triangle_indices[1], triangle.b);
  assert.equal(packed.triangle_indices[2], triangle.c);
  assert.equal(packed.triangle_confidence[0], triangle.confidence);
}

// Warm the result-boundary helpers independently from reconstruction.
const warmObject = objectGeometry(64, 96).result;
const warmPacked = packedGeometry(64, 96).result;
assertEquivalentGeometry(warmObject, warmPacked, 64, 96);

const resultBoundaryCases = [
  { pointCount: 1_000, triangleCount: 2_000 },
  { pointCount: 10_000, triangleCount: 20_000 },
  { pointCount: 50_000, triangleCount: 100_000 },
];

const outputCases = resultBoundaryCases.map(({ pointCount, triangleCount }) => {
  const objectRuns: number[] = [];
  const packedRuns: number[] = [];
  let objectResult: ObjectGeometryPayload | undefined;
  let packedResult: PackedGeometryPayload | undefined;

  for (let iteration = 0; iteration < 3; iteration += 1) {
    const first = iteration % 2 === 0 ? "object" : "packed";
    for (const representation of [first, first === "object" ? "packed" : "object"] as const) {
      if (representation === "object") {
        const measurement = objectGeometry(pointCount, triangleCount);
        objectRuns.push(measurement.elapsedMs);
        objectResult = measurement.result;
      } else {
        const measurement = packedGeometry(pointCount, triangleCount);
        packedRuns.push(measurement.elapsedMs);
        packedResult = measurement.result;
      }
    }
  }

  assert.ok(objectResult && packedResult);
  assertEquivalentGeometry(
    objectResult,
    packedResult,
    pointCount,
    triangleCount,
  );

  const objectMedianMs = median(objectRuns);
  const packedMedianMs = median(packedRuns);
  return {
    point_count: pointCount,
    triangle_count: triangleCount,
    packed_payload_bytes: pointCount * 19 + triangleCount * 16,
    object_graph_median_ms: objectMedianMs,
    packed_typed_arrays_median_ms: packedMedianMs,
    packed_speedup: objectMedianMs / packedMedianMs,
  };
});

console.log(
  JSON.stringify(
    {
      benchmark: "wasm-reconstruction-result-boundary",
      cases: outputCases,
      note: "Synthetic geometry isolates Rust-to-JavaScript serialization from reconstruction compute; timing is informational only.",
    },
    null,
    2,
  ),
);
