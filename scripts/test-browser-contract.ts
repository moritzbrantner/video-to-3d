import init, {
  assess_input_readiness,
  bake_textured_surface,
  reconstruct_sequence,
} from "../apps/web/public/wasm/video_to_3d_wasm.js";
import {
  assertReconstructionContract,
  normalizeWasmReconstruction,
} from "../apps/web/src/reconstruction";
import {
  buildSurfaceMaterialRequest,
  normalizeSurfaceMaterialResult,
  retainReferenceImages,
  type SurfaceMaterialSource,
} from "../apps/web/src/surfaceMaterials";

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
  result.dense.working_set_estimate.total_bytes <= 0 ||
  result.dense.working_set_budget_bytes !== 256 * 1024 * 1024
) {
  throw new Error("WASM reconstruction did not expose its Rust-owned dense working-set budget");
}

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

// Every adjacent pair crosses the boundary as a seed candidate, and an uncalibrated
// result names the decisive seed gate rather than only "no registered cameras".
if (
  result.seed_candidates.length !== 1 ||
  result.seed_candidates[0].rejected_gate === null ||
  result.bootstrap.decisive_gate !== result.seed_candidates[0].rejected_gate ||
  !result.bootstrap.summary.includes("decisive gate")
) {
  throw new Error(
    `bootstrap contract mismatch: ${JSON.stringify(result.bootstrap)} / ${JSON.stringify(result.seed_candidates)}`,
  );
}
const silentBootstrap = structuredClone(result);
silentBootstrap.bootstrap = { ...silentBootstrap.bootstrap, decisive_gate: null };
let silentBootstrapRejected = false;
try {
  assertReconstructionContract(silentBootstrap, 2);
} catch (error) {
  if (String(error).includes("bootstrap contract mismatch")) {
    silentBootstrapRejected = true;
  } else {
    throw error;
  }
}
if (!silentBootstrapRejected) {
  throw new Error("TypeScript accepted an uncalibrated result without a decisive bootstrap gate");
}

// Readiness evidence crosses the same boundary; flat duplicate frames must be
// reported as unsuitable for geometry while generative paths stay allowed.
const readinessValue = assess_input_readiness({
  frames: [
    { width, height, rgba },
    { width, height, rgba },
  ],
  metadata: {
    duration_seconds: 2,
    display_width: width,
    display_height: height,
    requested_times: [0.1, 1.1],
    presented_times: [0.1, 1.1],
  },
});
const readiness = (readinessValue instanceof Map
  ? Object.fromEntries(readinessValue)
  : readinessValue) as Record<string, unknown>;
if (readiness.geometric_verdict !== "unsuitable" || readiness.generative_paths_allowed !== true) {
  throw new Error(
    `readiness contract mismatch: ${String(readiness.geometric_verdict)} / ${String(readiness.generative_paths_allowed)}`,
  );
}

if (!Array.isArray(result.accepted_camera_evidence)) {
  throw new Error("WASM reconstruction did not expose accepted camera evidence for material export");
}

// Surface material export: two reference patches (frames 0 and 1) share one
// mixed triangle that must fall back; the GLB must load without browser state.
const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const patch = (reference_frame: number, source: number, start: number) => ({
  reference_frame,
  source_frames: [source],
  primary_start: start,
  primary_points: 3,
  completion_start: start + 3,
  completed_points: 0,
  sampled_pixels: 0,
  accepted_points: 3,
  reciprocal_consistent_points: 3,
  grid_stride: 4,
  grid_border: 0,
  search_min_depth: null,
  search_max_depth: null,
});
const materialSource: SurfaceMaterialSource = {
  dense_points: {
    values: new Float32Array([
      0, 0, 2, 0.9, 1, 0, 2, 0.9, 0, 1, 2, 0.9,
      2, 0, 2, 0.9, 3, 0, 2, 0.9, 2, 1, 2, 0.9,
    ]),
    rgb: new Uint8Array(18).fill(140),
    length: 6,
  },
  dense_grid_sites: {
    xy: new Uint32Array([2, 2, 8, 2, 2, 8, 3, 3, 9, 3, 3, 9]),
    length: 6,
  },
  mesh_triangles: {
    indices: new Uint32Array([0, 1, 2, 3, 4, 5, 1, 3, 2]),
    confidence: new Float32Array([0.8, 0.8, 0.8]),
    length: 3,
  },
  accepted_camera_evidence: [0, 1].map((frame_index) => ({
    frame_index,
    authority: "calibrated_seed" as const,
    rotation: identity,
    translation: [frame_index * 0.2, 0, 0],
    confidence: 0.9,
    median_reprojection_error_pixels: 0.4,
  })),
  dense: { reference_patches: [patch(0, 1, 0), patch(1, 0, 3)] },
};
const referenceImages = retainReferenceImages(materialSource, [
  { width, height, rgba },
  { width, height, rgba },
  { width, height, rgba },
]);
if (referenceImages.map((image) => image.frame_index).join() !== "0,1") {
  throw new Error("material export must retain exactly the accepted reference frames");
}
const baked = normalizeSurfaceMaterialResult(
  bake_textured_surface(buildSurfaceMaterialRequest(materialSource, referenceImages)),
);
if (
  baked.bake.materials.length !== 2 ||
  baked.bake.fallback.reasons.mixed_reference !== 1 ||
  !baked.diagnostic.includes("2 of 3 accepted triangles textured")
) {
  throw new Error(`surface material bake mismatch: ${baked.diagnostic}`);
}
const glb = new DataView(baked.glb.buffer, baked.glb.byteOffset, baked.glb.byteLength);
if (glb.getUint32(0, true) !== 0x46546c67 || glb.getUint32(8, true) !== baked.glb.byteLength) {
  throw new Error("textured surface export is not a well-formed GLB");
}
const jsonLength = glb.getUint32(12, true);
const gltf = JSON.parse(new TextDecoder().decode(baked.glb.subarray(20, 20 + jsonLength)));
const binOffset = 20 + jsonLength + 8;
const pngView = gltf.bufferViews[gltf.images[0].bufferView];
const pngSignature = baked.glb.subarray(binOffset + pngView.byteOffset, binOffset + pngView.byteOffset + 4);
if (
  gltf.images.length !== 2 ||
  gltf.images.some((image: { mimeType: string; uri?: string }) => image.mimeType !== "image/png" || image.uri) ||
  pngSignature[1] !== 0x50 ||
  gltf.materials[0].extras.video_to_3d.reference_frame !== 0 ||
  gltf.materials[0].extras.video_to_3d.appearance_key !== baked.appearance[0].appearance_key ||
  gltf.materials[2].extras.video_to_3d.fallback !== "vertex_color"
) {
  throw new Error("textured GLB does not carry embedded textures and reference provenance");
}
const rebaked = normalizeSurfaceMaterialResult(
  bake_textured_surface(
    buildSurfaceMaterialRequest(materialSource, referenceImages, baked.appearance),
  ),
);
if (rebaked.invalidation.reused.join() !== "0,1" || rebaked.invalidation.invalidated.length !== 0) {
  throw new Error("unchanged reference appearance must be reused, not invalidated");
}

console.log(
  "Rust -> WASM -> TypeScript camera-state, readiness, and surface-material contracts passed",
);
