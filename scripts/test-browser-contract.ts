import init, {
  reconstruct_sequence,
} from "../apps/web/public/wasm/video_to_3d_wasm.js";
import {
  assertReconstructionContract,
  type ReconstructionResult,
} from "../apps/web/src/reconstruction";

const width = 48;
const height = 32;
const rgba = new Array<number>(width * height * 4).fill(120);
for (let index = 3; index < rgba.length; index += 4) {
  rgba[index] = 255;
}

const wasmPath = new URL(
  "../apps/web/public/wasm/video_to_3d_wasm_bg.wasm",
  import.meta.url,
);
const wasmBytes = await Bun.file(wasmPath).arrayBuffer();
await init({ module_or_path: wasmBytes });

const result = reconstruct_sequence({
  frames: [
    { width, height, rgba },
    { width, height, rgba },
  ],
  options: {
    max_features: 320,
    min_feature_distance: 7,
    descriptor_radius: 3,
    match_radius: 42,
    max_descriptor_distance: 36,
    ratio_threshold: 0.82,
  },
}) as ReconstructionResult;

assertReconstructionContract(result, 2);

if (result.calibrated_pair !== null) {
  throw new Error("flat browser-contract fixture unexpectedly calibrated a seed pair");
}
if (result.camera_state.approximate_motion_samples.length !== 2) {
  throw new Error(
    `expected 2 approximate motion samples, got ${result.camera_state.approximate_motion_samples.length}`,
  );
}
if (
  result.camera_state.calibrated_seed_cameras.length !== 0 ||
  result.camera_state.registered_cameras.length !== 0 ||
  result.camera_state.dense_eligible_cameras.length !== 0
) {
  throw new Error("uncalibrated WASM result exposed accepted camera geometry");
}

console.log("Rust -> WASM -> TypeScript camera-state contract passed");
