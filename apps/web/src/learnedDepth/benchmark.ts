import type { ReconstructionResult, SampledFrame } from "../reconstruction";
import {
  learnedProviderDescriptor,
  learnedProvidersForMode,
  type LearnedProviderId,
  type LearnedReconstructionMode,
} from "./catalog";
import {
  evaluateRelativeDepthGeometry,
  type GeometryDepthFrameEvidence,
} from "./geometry";
import { evidenceQuality, median, percentile } from "./metrics";
import { recoverMogeDepth } from "./mogeGeometry";
import { createLearnedProviderSession } from "./runtime";
import { selectLearnedFrameIndices } from "./selection";
import type {
  LearnedBenchmarkSuite,
  LearnedGeometryBenchmark,
  LearnedProviderBenchmark,
} from "./types";

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarizeGeometry(
  evaluation: Awaited<ReturnType<typeof evaluateRelativeDepthGeometry>>,
): LearnedGeometryBenchmark | null {
  if (!evaluation) return null;
  const errors = evaluation.frames
    .map((frame) => frame.median_relative_error)
    .filter((value): value is number => value !== null && Number.isFinite(value));
  const agreement = evaluation.frames
    .map((frame) => frame.agreement_ratio_15_percent)
    .filter((value): value is number => value !== null && Number.isFinite(value));
  return {
    evaluatedFrames: evaluation.frames.length,
    usableFrames: evaluation.frames.filter((frame) => frame.calibration_usable).length,
    medianRelativeError: median(errors),
    meanAgreementRatio15Percent: mean(agreement),
  };
}

async function benchmarkProvider(
  providerId: LearnedProviderId,
  frames: SampledFrame[],
  selectedFrames: number[],
  reconstruction: ReconstructionResult,
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
      geometricAgreement: null,
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
    const geometryDepthEvidence: GeometryDepthFrameEvidence[] = [];
    let mogeRecoveredFrames = 0;
    let mogeRecoveryFailures = 0;
    const focalPixels = reconstruction.calibrated_pair?.focal_pixels ?? null;

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
      if (evidence.representation === "relative_depth") {
        geometryDepthEvidence.push(evidence);
      } else if (focalPixels && Number.isFinite(focalPixels) && focalPixels > 0) {
        try {
          const recovered = recoverMogeDepth(
            evidence.points,
            evidence.confidence,
            evidence.metricScale,
            evidence.letterbox,
            focalPixels,
          );
          geometryDepthEvidence.push({
            frameIndex: evidence.frameIndex,
            width: recovered.width,
            height: recovered.height,
            depth: recovered.depth,
          });
          mogeRecoveredFrames += 1;
        } catch {
          mogeRecoveryFailures += 1;
        }
      }
    }

    const geometryEvaluation =
      geometryDepthEvidence.length > 0
        ? await evaluateRelativeDepthGeometry(reconstruction, geometryDepthEvidence)
        : null;
    const geometricAgreement = summarizeGeometry(geometryEvaluation);
    let diagnostic: string;
    if (descriptor.evidence === "affine_point_map") {
      if (mogeRecoveredFrames > 0 && geometricAgreement) {
        diagnostic = `Recovered MoGe depth with the accepted SfM focal on ${mogeRecoveredFrames} frames and Rust found ${geometricAgreement.usableFrames}/${geometricAgreement.evaluatedFrames} geometrically usable; ${mogeRecoveryFailures} frame recoveries failed closed. No learned geometry is promoted yet.`;
      } else if (!focalPixels) {
        diagnostic = "MoGe produced affine point evidence, but no accepted classical focal was available for fail-closed Z-shift recovery.";
      } else {
        diagnostic = `MoGe inference completed, but calibrated Z-shift recovery produced no evaluable depth frames; ${mogeRecoveryFailures} recoveries failed closed.`;
      }
    } else if (geometricAgreement) {
      diagnostic = `Rust aligned relative depth to final SfM geometry on ${geometricAgreement.usableFrames}/${geometricAgreement.evaluatedFrames} evaluated frames; this is still diagnostic evidence, not promoted mesh geometry.`;
    } else {
      diagnostic = "Relative-depth inference completed, but accepted dense geometry was insufficient for Rust alignment diagnostics.";
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
      geometricAgreement,
      diagnostic,
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
      geometricAgreement: null,
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
    providers.push(await benchmarkProvider(providerId, frames, selectedFrames, reconstruction));
  }
  return {
    selectedFrames,
    generatedAt: new Date().toISOString(),
    providers,
  };
}
