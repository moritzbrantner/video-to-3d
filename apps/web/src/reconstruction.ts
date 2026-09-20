export type SampledFrame = {
  width: number;
  height: number;
  rgba: Uint8Array;
  thumbnail: string;
  time: number;
};

export type CameraPose = {
  frame_index: number;
  x: number;
  y: number;
  z: number;
  matched_features: number;
};

export type Point3 = {
  x: number;
  y: number;
  z: number;
  confidence: number;
  r: number;
  g: number;
  b: number;
};

export type PairStats = {
  from_frame: number;
  to_frame: number;
  features_from: number;
  features_to: number;
  matches: number;
  overlap_ratio: number;
  median_dx: number;
  median_dy: number;
  median_motion: number;
  median_parallax_residual: number;
  low_parallax: boolean;
};

export type CalibratedPairStats = {
  from_frame: number;
  to_frame: number;
  matches: number;
  inliers: number;
  inlier_ratio: number;
  focal_pixels: number;
  median_sampson_error_pixels: number;
  median_reprojection_error_pixels: number;
  median_triangulation_angle_degrees: number;
  relative_rotation: number[];
  translation_direction: number[];
};

export type RegistrationCandidateStats = {
  frame_index: number;
  seed_landmark_correspondences: number;
  pnp_ready: boolean;
};

export type RegisteredViewStats = {
  frame_index: number;
  correspondences: number;
  inliers: number;
  inlier_ratio: number;
  median_reprojection_error_pixels: number;
  rotation: number[];
  translation: number[];
  recovered_from_revisit: boolean;
};

export type RevisitCandidateStats = {
  from_frame: number;
  to_frame: number;
  matches: number;
  overlap_ratio: number;
};

export type RevisitRecoveryStats = {
  frame_index: number;
  source_frame_index: number;
  matches: number;
  correspondences: number;
  accepted: boolean;
  inliers: number;
  median_reprojection_error_pixels: number | null;
};

export type RevisitClosureStats = {
  frame_index: number;
  source_frame_index: number;
  matches: number;
  correspondences: number;
  accepted: boolean;
  inliers: number;
  median_reprojection_error_pixels: number | null;
  camera_center_delta_seed_baselines: number | null;
  rotation_delta_degrees: number | null;
};

export type RevisitStats = {
  evaluated_pairs: number;
  candidates: RevisitCandidateStats[];
  recoveries: RevisitRecoveryStats[];
  closures: RevisitClosureStats[];
};

export type NewLandmarkStats = {
  candidate_tracks: number;
  accepted_landmarks: number;
  supporting_observations: number;
  median_reprojection_error_pixels: number | null;
  median_triangulation_angle_degrees: number | null;
};

export type BundleAdjustmentStats = {
  attempted: boolean;
  accepted: boolean;
  iterations: number;
  observations: number;
  optimized_cameras: number;
  optimized_landmarks: number;
  initial_median_reprojection_error_pixels: number | null;
  final_median_reprojection_error_pixels: number | null;
  initial_rmse_reprojection_error_pixels: number | null;
  final_rmse_reprojection_error_pixels: number | null;
};

export type MultiViewStats = {
  keyframes: number[];
  track_count: number;
  tracks_three_plus: number;
  longest_track: number;
  observations: number;
  linked_pairs: number;
  registration_candidates: RegistrationCandidateStats[];
  new_landmarks: NewLandmarkStats;
  bundle_adjustment: BundleAdjustmentStats;
};

export type DenseReferenceAttemptStats = {
  reference_frame: number;
  attempted: boolean;
  accepted: boolean;
  skip_reason: string | null;
  sampled_pixels: number;
  accepted_points: number;
  reciprocal_checked_points: number;
  reciprocal_rejected_points: number;
  reciprocal_consistent_points: number;
  surface_completion_proposals: number;
  surface_completed_points: number;
  surface_completion_rejected_texture: number;
  surface_completion_rejected_cross_view: number;
  surface_completion_rejected_reciprocal: number;
  surface_completion_rejected_fusion: number;
  surface_completion_rejected_footprint: number;
};

export type DenseReferencePatchStats = {
  reference_frame: number;
  source_frames: number[];
  primary_start: number;
  primary_points: number;
  completion_start: number;
  completed_points: number;
  sampled_pixels: number;
  accepted_points: number;
  reciprocal_consistent_points: number;
  grid_stride: number;
  grid_border: number;
  search_min_depth: number | null;
  search_max_depth: number | null;
};

export type DenseWorkingSetEstimate = {
  frame_bytes: number;
  retry_frame_bytes: number;
  remapped_frame_bytes: number;
  luminance_bytes: number;
  dense_sample_bytes: number;
  topology_bytes: number;
  packed_output_bytes: number;
  total_bytes: number;
};

export type DenseStats = {
  attempted: boolean;
  skip_reason: string | null;
  reference_frame: number | null;
  source_views: number;
  source_frames: number[];
  sampled_pixels: number;
  depth_hypotheses: number;
  accepted_points: number;
  surface_completion_proposals: number;
  surface_completed_points: number;
  surface_completion_rejected_texture: number;
  surface_completion_rejected_cross_view: number;
  surface_completion_rejected_reciprocal: number;
  surface_completion_rejected_fusion: number;
  surface_completion_rejected_footprint: number;
  grid_stride: number;
  grid_border: number;
  reciprocal_checked_points: number;
  reciprocal_rejected_points: number;
  reciprocal_consistent_points: number;
  fusion_input_observations: number;
  fusion_rejected_observations: number;
  fusion_rejected_points: number;
  median_fusion_observations: number | null;
  median_supporting_views: number | null;
  median_photometric_error: number | null;
  search_min_depth: number | null;
  search_max_depth: number | null;
  reference_frames: number[];
  reference_attempts: DenseReferenceAttemptStats[];
  reference_patches: DenseReferencePatchStats[];
  working_set_estimate: DenseWorkingSetEstimate;
  working_set_budget_bytes: number;
};

export type MeshTriangle = {
  a: number;
  b: number;
  c: number;
  confidence: number;
};

export type MeshReferencePatchStats = {
  reference_frame: number;
  point_count: number;
  attempted: boolean;
  skip_reason: string | null;
  grid_vertices: number;
  rejected_grid_vertices: number;
  candidate_cells: number;
  candidate_triangles: number;
  accepted_triangles: number;
  rejected_discontinuities: number;
  rejected_degenerate: number;
};

export type MeshStats = {
  attempted: boolean;
  skip_reason: string | null;
  reference_frame: number | null;
  grid_vertices: number;
  rejected_grid_vertices: number;
  candidate_cells: number;
  candidate_triangles: number;
  accepted_triangles: number;
  rejected_discontinuities: number;
  rejected_degenerate: number;
  reference_frames: number[];
  reference_patches: MeshReferencePatchStats[];
};

export type CameraKind = "none" | "approximate_motion" | "seed" | "registered";
export type RegistrationStatus =
  | "seed"
  | "registered"
  | "insufficient_correspondences"
  | "pnp_rejected"
  | "low_parallax"
  | "not_selected";
export type BundleAdjustmentStatus = "not_run" | "accepted" | "rejected";
export type DenseCameraRole = "reference" | "source";

export type FrameCameraState = {
  frame_index: number;
  camera_kind: CameraKind;
  registration_status: RegistrationStatus;
  correspondences: number;
  inliers: number;
  median_reprojection_error_pixels: number | null;
  recovered_from_revisit: boolean;
  bundle_adjustment: BundleAdjustmentStatus;
  dense_role: DenseCameraRole | null;
  dense_ineligibility_reason: string | null;
};

export type CameraPipelineState = {
  approximate_motion_samples: CameraPose[];
  calibrated_seed_cameras: CameraPose[];
  registered_cameras: CameraPose[];
  dense_eligible_cameras: CameraPose[];
  frames: FrameCameraState[];
};

export type DensePointBuffer = {
  values: Float32Array;
  rgb: Uint8Array;
  length: number;
};

export type DenseGridSiteBuffer = {
  xy: Uint32Array;
  length: number;
};

export type MeshTriangleBuffer = {
  indices: Uint32Array;
  confidence: Float32Array;
  length: number;
};

export type ReconstructionResult = {
  cameras: CameraPose[];
  points: Point3[];
  dense_points: DensePointBuffer;
  dense_grid_sites: DenseGridSiteBuffer;
  dense: DenseStats;
  mesh_triangles: MeshTriangleBuffer;
  mesh: MeshStats;
  pairs: PairStats[];
  calibrated_pair: CalibratedPairStats | null;
  multi_view: MultiViewStats;
  revisits: RevisitStats;
  registered_views: RegisteredViewStats[];
  warnings: string[];
  camera_state: CameraPipelineState;
};

type WasmModule = {
  default: () => Promise<unknown>;
  reconstruct_sequence: (request: unknown) => unknown;
};

type RawDensePointBuffer = {
  values_f32_le: Uint8Array;
  rgb: Uint8Array;
  length: number;
};

type RawDenseGridSiteBuffer = {
  xy_u32_le: Uint8Array;
  length: number;
};

type RawMeshTriangleBuffer = {
  indices_u32_le: Uint8Array;
  confidence_f32_le: Uint8Array;
  length: number;
};

type RawReconstructionResult = Omit<
  ReconstructionResult,
  "dense_points" | "dense_grid_sites" | "mesh_triangles"
> & {
  dense_points: RawDensePointBuffer;
  dense_grid_sites: RawDenseGridSiteBuffer;
  mesh_triangles: RawMeshTriangleBuffer;
};

const FEWER_THAN_TWO_REGISTERED_CAMERAS =
  "fewer than two accepted registered cameras are available";

const NATIVE_LITTLE_ENDIAN =
  new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;

let wasmPromise: Promise<WasmModule> | null = null;

async function loadWasm(): Promise<WasmModule> {
  if (!wasmPromise) {
    const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    const moduleUrl = `${basePath}/wasm/video_to_3d_wasm.js`;
    const importModule = new Function("url", "return import(url)") as (
      url: string,
    ) => Promise<WasmModule>;
    wasmPromise = importModule(moduleUrl).then(async (module) => {
      await module.default();
      return module;
    });
  }
  return wasmPromise;
}

function objectRecord(value: unknown, label: string): Record<string, unknown> {
  const normalized = value instanceof Map ? Object.fromEntries(value) : value;
  if (!normalized || typeof normalized !== "object") {
    throw new Error(`camera-state contract mismatch: WASM returned invalid ${label}`);
  }
  return normalized as Record<string, unknown>;
}

function byteBuffer(value: unknown, label: string): Uint8Array {
  if (!(value instanceof Uint8Array)) {
    throw new Error(`camera-state contract mismatch: ${label} is not a Uint8Array`);
  }
  return value;
}

function packedLength(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`camera-state contract mismatch: ${label} is not a valid length`);
  }
  return value as number;
}

function float32LittleEndian(bytes: Uint8Array, label: string): Float32Array {
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`camera-state contract mismatch: ${label} byte length is not divisible by 4`);
  }
  if (NATIVE_LITTLE_ENDIAN) {
    const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
    return new Float32Array(
      aligned.buffer,
      aligned.byteOffset,
      aligned.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
  }
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(bytes.byteLength / 4);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = source.getFloat32(index * 4, true);
  }
  return values;
}

function uint32LittleEndian(bytes: Uint8Array, label: string): Uint32Array {
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`camera-state contract mismatch: ${label} byte length is not divisible by 4`);
  }
  if (NATIVE_LITTLE_ENDIAN) {
    const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
    return new Uint32Array(
      aligned.buffer,
      aligned.byteOffset,
      aligned.byteLength / Uint32Array.BYTES_PER_ELEMENT,
    );
  }
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Uint32Array(bytes.byteLength / 4);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = source.getUint32(index * 4, true);
  }
  return values;
}

function normalizeDensePointBuffer(value: unknown): DensePointBuffer {
  const raw = objectRecord(value, "dense-point buffer") as RawDensePointBuffer;
  const length = packedLength(raw.length, "dense-point count");
  const values = float32LittleEndian(
    byteBuffer(raw.values_f32_le, "dense-point float buffer"),
    "dense-point float buffer",
  );
  const rgb = byteBuffer(raw.rgb, "dense-point RGB buffer");
  if (values.length !== length * 4 || rgb.length !== length * 3) {
    throw new Error("camera-state contract mismatch: dense-point buffer lengths are inconsistent");
  }
  return { values, rgb, length };
}

function normalizeDenseGridSiteBuffer(value: unknown): DenseGridSiteBuffer {
  const raw = objectRecord(value, "dense-grid-site buffer") as RawDenseGridSiteBuffer;
  const length = packedLength(raw.length, "dense-grid-site count");
  const xy = uint32LittleEndian(
    byteBuffer(raw.xy_u32_le, "dense-grid-site coordinate buffer"),
    "dense-grid-site coordinate buffer",
  );
  if (xy.length !== length * 2) {
    throw new Error("camera-state contract mismatch: dense-grid-site buffer lengths are inconsistent");
  }
  return { xy, length };
}

function normalizeMeshTriangleBuffer(value: unknown): MeshTriangleBuffer {
  const raw = objectRecord(value, "mesh-triangle buffer") as RawMeshTriangleBuffer;
  const length = packedLength(raw.length, "mesh-triangle count");
  const indices = uint32LittleEndian(
    byteBuffer(raw.indices_u32_le, "mesh-triangle index buffer"),
    "mesh-triangle index buffer",
  );
  const confidence = float32LittleEndian(
    byteBuffer(raw.confidence_f32_le, "mesh-triangle confidence buffer"),
    "mesh-triangle confidence buffer",
  );
  if (indices.length !== length * 3 || confidence.length !== length) {
    throw new Error("camera-state contract mismatch: mesh-triangle buffer lengths are inconsistent");
  }
  return { indices, confidence, length };
}

export function normalizeWasmReconstruction(value: unknown): ReconstructionResult {
  const normalized = objectRecord(value, "reconstruction") as RawReconstructionResult;
  return {
    ...normalized,
    dense_points: normalizeDensePointBuffer(normalized.dense_points),
    dense_grid_sites: normalizeDenseGridSiteBuffer(normalized.dense_grid_sites),
    mesh_triangles: normalizeMeshTriangleBuffer(normalized.mesh_triangles),
  };
}

function frameIndexSet(cameras: CameraPose[]): Set<number> {
  return new Set(cameras.map((camera) => camera.frame_index));
}

function sameFrameSet(left: Set<number>, right: Set<number>): boolean {
  return left.size === right.size && [...left].every((frameIndex) => right.has(frameIndex));
}

export function assertReconstructionContract(
  result: ReconstructionResult,
  expectedFrameCount: number,
): void {
  const state = result.camera_state;
  const workingSet = result.dense.working_set_estimate;
  if (!state || !Array.isArray(state.frames)) {
    throw new Error("camera-state contract mismatch: WASM result has no explicit camera_state");
  }
  if (
    !workingSet ||
    !Object.values(workingSet).every(
      (value) => Number.isSafeInteger(value) && value >= 0,
    ) ||
    !Number.isSafeInteger(result.dense.working_set_budget_bytes) ||
    result.dense.working_set_budget_bytes <= 0
  ) {
    throw new Error("camera-state contract mismatch: dense working-set evidence is invalid");
  }
  const workingSetSum =
    workingSet.frame_bytes +
    workingSet.retry_frame_bytes +
    workingSet.remapped_frame_bytes +
    workingSet.luminance_bytes +
    workingSet.dense_sample_bytes +
    workingSet.topology_bytes +
    workingSet.packed_output_bytes;
  if (workingSet.total_bytes !== workingSetSum) {
    throw new Error("camera-state contract mismatch: dense working-set components do not sum to the total");
  }
  if (
    workingSet.total_bytes > result.dense.working_set_budget_bytes &&
    (result.dense.attempted ||
      result.dense_points.length > 0 ||
      result.mesh.attempted ||
      result.mesh_triangles.length > 0 ||
      !result.dense.skip_reason?.includes("working set"))
  ) {
    throw new Error(
      "camera-state contract mismatch: over-budget dense reconstruction did not fail closed",
    );
  }
  if (
    result.dense_points.values.length !== result.dense_points.length * 4 ||
    result.dense_points.rgb.length !== result.dense_points.length * 3 ||
    result.dense_grid_sites.xy.length !== result.dense_grid_sites.length * 2 ||
    result.dense_grid_sites.length !== result.dense_points.length ||
    result.mesh_triangles.indices.length !== result.mesh_triangles.length * 3 ||
    result.mesh_triangles.confidence.length !== result.mesh_triangles.length
  ) {
    throw new Error("camera-state contract mismatch: packed reconstruction geometry is inconsistent");
  }
  if (
    !Array.isArray(result.dense.reference_frames) ||
    !Array.isArray(result.dense.reference_attempts) ||
    !Array.isArray(result.dense.reference_patches) ||
    !Array.isArray(result.mesh.reference_frames) ||
    !Array.isArray(result.mesh.reference_patches)
  ) {
    throw new Error(
      "camera-state contract mismatch: WASM result has no explicit multi-reference reconstruction evidence",
    );
  }
  if (
    result.dense.reference_attempts.some(
      (attempt) => !attempt.accepted && attempt.skip_reason === null,
    )
  ) {
    throw new Error(
      "camera-state contract mismatch: rejected dense reference attempt has no diagnostic reason",
    );
  }
  const acceptedCameras = [
    ...state.calibrated_seed_cameras,
    ...state.registered_cameras,
  ];
  const acceptedFrames = frameIndexSet(acceptedCameras);

  if (state.frames.length !== expectedFrameCount) {
    throw new Error(
      `camera-state contract mismatch: expected ${expectedFrameCount} frame states, got ${state.frames.length}`,
    );
  }
  if (state.frames.some((frame, index) => frame.frame_index !== index)) {
    throw new Error("camera-state contract mismatch: frame states are not index-aligned");
  }

  if (result.calibrated_pair) {
    if (state.calibrated_seed_cameras.length !== 2) {
      throw new Error(
        `camera-state contract mismatch: calibrated geometry exposed ${state.calibrated_seed_cameras.length} seed cameras instead of 2`,
      );
    }
    if (state.approximate_motion_samples.length !== 0) {
      throw new Error(
        "camera-state contract mismatch: approximate motion samples were exposed as calibrated camera geometry",
      );
    }
    if (acceptedCameras.length !== result.cameras.length) {
      throw new Error(
        "camera-state contract mismatch: explicit accepted-camera partition differs from the Rust reconstruction camera list",
      );
    }
    if (state.registered_cameras.length !== result.registered_views.length) {
      throw new Error(
        "camera-state contract mismatch: registered camera poses differ from registered-view evidence",
      );
    }
  } else {
    if (
      state.calibrated_seed_cameras.length !== 0 ||
      state.registered_cameras.length !== 0 ||
      state.dense_eligible_cameras.length !== 0
    ) {
      throw new Error(
        "camera-state contract mismatch: uncalibrated reconstruction exposed accepted camera geometry",
      );
    }
    if (state.approximate_motion_samples.length !== result.cameras.length) {
      throw new Error(
        "camera-state contract mismatch: approximate motion samples differ from the fallback motion track",
      );
    }
  }

  if (result.dense.source_frames.length !== result.dense.source_views) {
    throw new Error(
      `camera-state contract mismatch: dense reports ${result.dense.source_views} source views but ${result.dense.source_frames.length} source frame ids`,
    );
  }
  if (new Set(result.dense.source_frames).size !== result.dense.source_frames.length) {
    throw new Error("camera-state contract mismatch: dense source frame ids are not unique");
  }

  if (
    result.dense.skip_reason?.includes(FEWER_THAN_TWO_REGISTERED_CAMERAS) &&
    acceptedCameras.length >= 2
  ) {
    throw new Error(
      `camera-state contract mismatch: dense reports fewer than two accepted registered cameras while ${acceptedCameras.length} accepted cameras are exposed`,
    );
  }

  if (result.dense.attempted) {
    if (result.dense.reference_frame === null) {
      throw new Error(
        "camera-state contract mismatch: attempted dense reconstruction has no reference frame",
      );
    }
    if (acceptedCameras.length < 2) {
      throw new Error(
        `camera-state contract mismatch: dense reconstruction ran with only ${acceptedCameras.length} accepted cameras`,
      );
    }
    const expectedDenseFrames = new Set([
      result.dense.reference_frame,
      ...result.dense.source_frames,
    ]);
    const actualDenseFrames = frameIndexSet(state.dense_eligible_cameras);
    if (!sameFrameSet(expectedDenseFrames, actualDenseFrames)) {
      throw new Error(
        "camera-state contract mismatch: dense reference/source frames differ from dense-eligible camera state",
      );
    }
    if ([...actualDenseFrames].some((frameIndex) => !acceptedFrames.has(frameIndex))) {
      throw new Error(
        "camera-state contract mismatch: dense-eligible camera is not part of accepted camera geometry",
      );
    }
  } else if (state.dense_eligible_cameras.length !== 0) {
    throw new Error(
      "camera-state contract mismatch: skipped dense reconstruction exposed dense-eligible cameras",
    );
  }
}

export async function reconstructFrames(frames: SampledFrame[]): Promise<ReconstructionResult> {
  const wasm = await loadWasm();
  const result = normalizeWasmReconstruction(
    wasm.reconstruct_sequence({
      frames: frames.map(({ width, height, rgba }) => ({ width, height, rgba })),
      options: {
        max_features: 320,
        min_feature_distance: 7,
        descriptor_radius: 3,
        match_radius: 42,
        max_descriptor_distance: 36,
        ratio_threshold: 0.82,
        max_dense_working_set_bytes: 256 * 1024 * 1024,
      },
    }),
  );
  assertReconstructionContract(result, frames.length);
  return result;
}
