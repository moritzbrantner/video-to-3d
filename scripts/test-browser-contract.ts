import init, {
  reconstruct_sequence,
} from "../apps/web/public/wasm/video_to_3d_wasm.js";
import {
  assertReconstructionContract,
  normalizeWasmReconstruction,
} from "../apps/web/src/reconstruction";

const width = 48;
const height = 32;
const rgba = new Uint8Array(width * height * 4);
rgba.fill(120);
for (let index = 3; index < rgba.length; index += 4) {
  rgba[index] = 255;
}

const wasmPath = new URL(
  "../apps/web/public/wasm/video_to_3d_wasm_bg.wasm",
  import.meta.url,
);
const wasmBytes = await Bun.file(wasmPath).arrayBuffer();
await init({ module_or_path: wasmBytes });

const result = normalizeWasmReconstruction(
  reconstruct_sequence({
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
  }),
);

assertReconstructionContract(result, 2);

if (
  !(result.dense_points.values instanceof Float32Array) ||
  !(result.dense_points.rgb instanceof Uint8Array) ||
  !(result.mesh_triangles.indices instanceof Uint32Array) ||
  !(result.mesh_triangles.confidence instanceof Float32Array)
) {
  throw new Error("WASM reconstruction did not expose packed typed geometry buffers");
}
if (result.dense_points.length !== 0 || result.mesh_triangles.length !== 0) {
  throw new Error("flat browser-contract fixture unexpectedly exposed reconstructed geometry");
}

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

const [firstSeed, secondSeed] = result.camera_state.approximate_motion_samples;
const contradictory = structuredClone(result);
contradictory.calibrated_pair = {
  from_frame: 0,
  to_frame: 1,
  matches: 8,
  inliers: 8,
  inlier_ratio: 1,
  focal_pixels: width,
  median_sampson_error_pixels: 0,
  median_reprojection_error_pixels: 0,
  median_triangulation_angle_degrees: 1,
  relative_rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  translation_direction: [1, 0, 0],
};
contradictory.cameras = [firstSeed, secondSeed];
contradictory.camera_state = {
  approximate_motion_samples: [],
  calibrated_seed_cameras: [firstSeed, secondSeed],
  registered_cameras: [],
  dense_eligible_cameras: [],
  frames: contradictory.camera_state.frames.map((frame) => ({
    ...frame,
    camera_kind: "seed",
    registration_status: "seed",
  })),
};
contradictory.dense = {
  ...contradictory.dense,
  attempted: false,
  reference_frame: null,
  source_views: 0,
  source_frames: [],
  skip_reason: "fewer than two accepted registered cameras are available",
};

let contradictionRejected = false;
try {
  assertReconstructionContract(contradictory, 2);
} catch (error) {
  if (String(error).includes("fewer than two accepted registered cameras")) {
    contradictionRejected = true;
  } else {
    throw error;
  }
}
if (!contradictionRejected) {
  throw new Error(
    "TypeScript accepted contradictory dense camera evidence after Rust/WASM serialization",
  );
}

console.log("Rust -> WASM -> TypeScript camera-state contract passed");
