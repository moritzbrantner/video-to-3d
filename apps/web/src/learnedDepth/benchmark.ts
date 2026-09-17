import type { ReconstructionResult, SampledFrame } from "../reconstruction";
import {
  learnedProviderDescriptor,
  learnedProvidersForMode,
  type LearnedProviderId,
  type LearnedReconstructionMode,
} from "./catalog";
import { evidenceQuality, median, percentile } from "./metrics";
import { createLearnedProviderSession } from "./runtime";
import { selectLearnedFrameIndices } from "./selection";
import type { LearnedBenchmarkSuite, LearnedProviderBenchmark } from "./types";

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

async function benchmarkProvider(
  providerId: LearnedProviderId,
  frames: SampledFrame[],
  selectedFrames: number[],
): Promise<LearnedProviderBenchmark> {
  const descriptor = learnedProviderDescriptor(providerId);
  if (selectedFrames.length === 0) {
    return {
      providerId,
      label: descriptor.label,
      status: "skipped",
      backend: null,
      runtimeLabel: null,
      modelReference: descriptor.modelReference,
      approximateDownloadBytes: descriptor.approximateDownloadBytes,
      selectedFrames,
      loadMs: null,
      inferenceMs: [],
      medianInferenceMs: null,
      p90InferenceMs: null,
      finiteEvidenceRatio: null,
      confidenceCoverage: null,
      diagnostic: "Skipped because the classical reconstruction exposed no accepted registered cameras.",
    };
  }

  const loadStartedAt = performance.now();
  let session: Awaited<ReturnType<typeof createLearnedProviderSession>> | null = null;
  try {
    session = await createLearnedProviderSession(providerId);
    const loadMs = performance.now() - loadStartedAt;
    const inferenceMs: number[] = [];
    const finiteRatios: number[] = [];
    const confidenceCoverage: number[] = [];

    for (const frameIndex of selectedFrames) {
      const frame = frames[frameIndex];
      if (!frame) throw new Error(`selected learned frame ${frameIndex} is missing`);
      const evidence = await session.infer(frame, frameIndex);
      inferenceMs.push(evidence.inferenceMs);
      const quality = evidenceQuality(evidence);
      finiteRatios.push(quality.finiteRatio);
      if (quality.confidenceCoverage !== null) {
        confidenceCoverage.push(quality.confidenceCoverage);
      }
    }

    return {
      providerId,
      label: descriptor.label,
      status: "completed",
      backend: session.backend,
      runtimeLabel: session.runtimeLabel,
      modelReference: descriptor.modelReference,
      approximateDownloadBytes: descriptor.approximateDownloadBytes,
      selectedFrames,
      loadMs,
      inferenceMs,
      medianInferenceMs: median(inferenceMs),
      p90InferenceMs: percentile(inferenceMs, 0.9),
      finiteEvidenceRatio: mean(finiteRatios),
      confidenceCoverage: mean(confidenceCoverage),
      diagnostic:
        descriptor.evidence === "affine_point_map"
          ? "Raw MoGe affine point evidence only; focal/shift recovery and Rust cross-view validation are still required before geometry promotion."
          : "Relative-depth evidence only; Rust scale alignment and cross-view validation are still required before geometry promotion.",
    };
  } catch (error) {
    return {
      providerId,
      label: descriptor.label,
      status: "failed",
      backend: session?.backend ?? null,
      runtimeLabel: session?.runtimeLabel ?? null,
      modelReference: descriptor.modelReference,
      approximateDownloadBytes: descriptor.approximateDownloadBytes,
      selectedFrames,
      loadMs: null,
      inferenceMs: [],
      medianInferenceMs: null,
      p90InferenceMs: null,
      finiteEvidenceRatio: null,
      confidenceCoverage: null,
      diagnostic: error instanceof Error ? error.message : String(error),
    };
  } finally {
    await session?.dispose();
  }
}

export async function benchmarkLearnedMode(
  mode: LearnedReconstructionMode,
  frames: SampledFrame[],
  reconstruction: ReconstructionResult,
): Promise<LearnedBenchmarkSuite | null> {
  const providerIds = learnedProvidersForMode(mode);
  if (providerIds.length === 0) return null;

  const selectedFrames = selectLearnedFrameIndices(reconstruction);
  const providers: LearnedProviderBenchmark[] = [];
  for (const providerId of providerIds) {
    providers.push(await benchmarkProvider(providerId, frames, selectedFrames));
  }
  return {
    selectedFrames,
    generatedAt: new Date().toISOString(),
    providers,
  };
}
