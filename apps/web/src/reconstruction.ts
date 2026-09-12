export type SampledFrame = {
  width: number;
  height: number;
  rgba: number[];
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
};

export type MeshTriangle = {
  a: number;
  b: number;
  c: number;
  confidence: number;
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

export type ReconstructionResult = {
  cameras: CameraPose[];
  points: Point3[];
  dense_points: Point3[];
  dense: DenseStats;
  mesh_triangles: MeshTriangle[];
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
  reconstruct_sequence: (request: unknown) => ReconstructionResult;
};

const FEWER_THAN_TWO_REGISTERED_CAMERAS =
  "fewer than two accepted registered cameras are available";

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
  const result = wasm.reconstruct_sequence({
    frames: frames.map(({ width, height, rgba }) => ({ width, height, rgba })),
    options: {
      max_features: 320,
      min_feature_distance: 7,
      descriptor_radius: 3,
      match_radius: 42,
      max_descriptor_distance: 36,
      ratio_threshold: 0.82,
    },
  });
  assertReconstructionContract(result, frames.length);
  return result;
}
