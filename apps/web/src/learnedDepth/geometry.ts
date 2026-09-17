import type { ReconstructionResult } from "../reconstruction";

export type GeometryDepthFrameEvidence = {
  frameIndex: number;
  width: number;
  height: number;
  depth: Float32Array;
};

export type LearnedGeometryFrameDiagnostic = {
  frame_index: number;
  projected_geometry_points: number;
  valid_alignment_anchors: number;
  fit_kind: "direct" | "inverse" | null;
  scale: number | null;
  offset: number | null;
  median_relative_error: number | null;
  agreement_ratio_15_percent: number | null;
  calibration_usable: boolean;
  skip_reason: string | null;
};

export type LearnedGeometryEvaluation = {
  agreement_threshold: number;
  frames: LearnedGeometryFrameDiagnostic[];
};

type AcceptedCameraEvidence = {
  frame_index: number;
  rotation: number[];
  translation: number[];
};

type ReconstructionWithEvidence = ReconstructionResult & {
  accepted_camera_evidence?: AcceptedCameraEvidence[];
};

type LearnedWasmModule = {
  default: () => Promise<unknown>;
  evaluate_relative_depth_evidence: (request: unknown) => unknown;
};

const NATIVE_LITTLE_ENDIAN =
  new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;
let wasmPromise: Promise<LearnedWasmModule> | null = null;

async function loadLearnedWasm(): Promise<LearnedWasmModule> {
  if (!wasmPromise) {
    const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    const moduleUrl = `${basePath}/wasm/video_to_3d_wasm.js`;
    const importModule = new Function("url", "return import(url)") as (
      url: string,
    ) => Promise<LearnedWasmModule>;
    wasmPromise = importModule(moduleUrl).then(async (module) => {
      await module.default();
      return module;
    });
  }
  return wasmPromise;
}

function float32LittleEndianBytes(values: Float32Array): Uint8Array {
  if (NATIVE_LITTLE_ENDIAN) {
    return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  }
  const bytes = new Uint8Array(values.byteLength);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setFloat32(index * Float32Array.BYTES_PER_ELEMENT, values[index], true);
  }
  return bytes;
}

function normalizeEvaluation(value: unknown): LearnedGeometryEvaluation {
  if (!value || typeof value !== "object") {
    throw new Error("learned-depth geometry evaluator returned an invalid result");
  }
  const evaluation = value as LearnedGeometryEvaluation;
  if (!Number.isFinite(evaluation.agreement_threshold) || !Array.isArray(evaluation.frames)) {
    throw new Error("learned-depth geometry evaluator returned an invalid contract");
  }
  return evaluation;
}

export async function evaluateRelativeDepthGeometry(
  reconstruction: ReconstructionResult,
  evidence: GeometryDepthFrameEvidence[],
): Promise<LearnedGeometryEvaluation | null> {
  if (evidence.length === 0 || reconstruction.dense_points.length === 0) return null;
  const focalPixels = reconstruction.calibrated_pair?.focal_pixels;
  if (!focalPixels || !Number.isFinite(focalPixels) || focalPixels <= 0) return null;

  const cameras = (reconstruction as ReconstructionWithEvidence).accepted_camera_evidence;
  if (!Array.isArray(cameras) || cameras.length < 2) {
    throw new Error("learned-depth geometry evaluation requires final accepted camera evidence");
  }
  for (const camera of cameras) {
    if (camera.rotation.length !== 9 || camera.translation.length !== 3) {
      throw new Error("learned-depth geometry evaluation received an invalid final camera pose");
    }
  }

  const wasm = await loadLearnedWasm();
  const result = wasm.evaluate_relative_depth_evidence({
    focal_pixels: focalPixels,
    cameras: cameras.map(({ frame_index, rotation, translation }) => ({
      frame_index,
      rotation,
      translation,
    })),
    dense_point_count: reconstruction.dense_points.length,
    dense_points_f32_le: float32LittleEndianBytes(reconstruction.dense_points.values),
    frames: evidence.map((frame) => ({
      frame_index: frame.frameIndex,
      width: frame.width,
      height: frame.height,
      values_f32_le: float32LittleEndianBytes(frame.depth),
    })),
  });
  return normalizeEvaluation(result);
}
