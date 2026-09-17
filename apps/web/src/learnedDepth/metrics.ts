import type { LearnedFrameEvidence } from "./types";

export function percentile(values: number[], fraction: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const clamped = Math.min(1, Math.max(0, fraction));
  const index = Math.ceil(clamped * sorted.length) - 1;
  return sorted[Math.max(0, index)];
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

export function evidenceQuality(evidence: LearnedFrameEvidence): {
  finiteRatio: number;
  confidenceCoverage: number | null;
} {
  if (evidence.representation === "relative_depth") {
    let finite = 0;
    for (const value of evidence.depth) {
      if (Number.isFinite(value)) finite += 1;
    }
    return {
      finiteRatio: evidence.depth.length === 0 ? 0 : finite / evidence.depth.length,
      confidenceCoverage: null,
    };
  }

  let finite = 0;
  const pointCount = Math.floor(evidence.points.length / 3);
  for (let index = 0; index < pointCount; index += 1) {
    const offset = index * 3;
    if (
      Number.isFinite(evidence.points[offset]) &&
      Number.isFinite(evidence.points[offset + 1]) &&
      Number.isFinite(evidence.points[offset + 2])
    ) {
      finite += 1;
    }
  }
  let confident = 0;
  for (const value of evidence.confidence) {
    if (Number.isFinite(value) && value > 0.5) confident += 1;
  }
  return {
    finiteRatio: pointCount === 0 ? 0 : finite / pointCount,
    confidenceCoverage:
      evidence.confidence.length === 0 ? 0 : confident / evidence.confidence.length,
  };
}
