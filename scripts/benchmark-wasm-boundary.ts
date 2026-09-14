import assert from "node:assert/strict";
import init, {
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

function reconstruct(frames: unknown[]): { elapsedMs: number; result: unknown } {
  const started = performance.now();
  const result = normalizeWasmReconstruction(
    reconstruct_sequence({ frames, options }),
  );
  return { elapsedMs: performance.now() - started, result };
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
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
      benchmark: "wasm-reconstruction-boundary",
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
