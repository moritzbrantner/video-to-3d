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

export type RevisitStats = {
  candidates: RevisitCandidateStats[];
  recoveries: RevisitRecoveryStats[];
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

export type ReconstructionResult = {
  cameras: CameraPose[];
  points: Point3[];
  pairs: PairStats[];
  calibrated_pair: CalibratedPairStats | null;
  multi_view: MultiViewStats;
  revisits: RevisitStats;
  registered_views: RegisteredViewStats[];
  warnings: string[];
};

type WasmModule = {
  default: () => Promise<unknown>;
  reconstruct_sequence: (request: unknown) => ReconstructionResult;
};

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

export async function reconstructFrames(frames: SampledFrame[]): Promise<ReconstructionResult> {
  const wasm = await loadWasm();
  return wasm.reconstruct_sequence({
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
}
