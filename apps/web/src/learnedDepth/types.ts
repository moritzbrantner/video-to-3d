import type { SampledFrame } from "../reconstruction";
import type { LearnedProviderDescriptor, LearnedProviderId } from "./catalog";

export type BrowserInferenceBackend = "webgpu" | "wasm";

export type RelativeDepthFrameEvidence = {
  providerId: "depth-anything-v2-small";
  frameIndex: number;
  representation: "relative_depth";
  width: number;
  height: number;
  depth: Float32Array;
  inferenceMs: number;
};

export type AffinePointMapFrameEvidence = {
  providerId: "moge-2-vits";
  frameIndex: number;
  representation: "affine_point_map";
  width: number;
  height: number;
  points: Float32Array;
  confidence: Float32Array;
  metricScale: number | null;
  inferenceMs: number;
};

export type LearnedFrameEvidence = RelativeDepthFrameEvidence | AffinePointMapFrameEvidence;

export type LearnedProviderSession = {
  descriptor: LearnedProviderDescriptor;
  backend: BrowserInferenceBackend;
  runtimeLabel: string;
  infer(frame: SampledFrame, frameIndex: number): Promise<LearnedFrameEvidence>;
  dispose(): Promise<void> | void;
};

export type LearnedGeometryBenchmark = {
  evaluatedFrames: number;
  usableFrames: number;
  medianRelativeError: number | null;
  meanAgreementRatio15Percent: number | null;
};

export type LearnedProviderBenchmark = {
  providerId: LearnedProviderId;
  label: string;
  status: "completed" | "failed" | "skipped";
  backend: BrowserInferenceBackend | null;
  runtimeLabel: string | null;
  modelReference: string;
  approximateDownloadBytes: number;
  selectedFrames: number[];
  loadMs: number | null;
  inferenceMs: number[];
  medianInferenceMs: number | null;
  p90InferenceMs: number | null;
  finiteEvidenceRatio: number | null;
  confidenceCoverage: number | null;
  geometricAgreement: LearnedGeometryBenchmark | null;
  diagnostic: string;
};

export type LearnedBenchmarkSuite = {
  selectedFrames: number[];
  generatedAt: string;
  providers: LearnedProviderBenchmark[];
};
