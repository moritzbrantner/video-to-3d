export type VideoSamplingOptions = {
  framesPerSecond?: number;
  maxFrames?: number;
};

export type VideoSamplingMetadata = {
  duration: number;
  videoWidth: number;
  videoHeight: number;
};

export type VideoSamplingPlan = {
  framesPerSecond: number;
  maxFrames: number;
  sampleCount: number;
  analysisWidth: number;
  analysisHeight: number;
  times: number[];
};

/**
 * Without an explicit frame cap, long clips are sampled at least this densely so
 * consecutive samples keep enough view overlap to chain cameras (#230); the Rust
 * keyframe selection then drops redundant samples by measured overlap. Short clips
 * keep the 18-frame default.
 */
export const MAX_AUTO_SAMPLE_SPACING_SECONDS = 1.0;
const DEFAULT_MAX_FRAMES = 18;
const MAX_SAMPLED_FRAMES = 120;

function positiveFinite(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : fallback;
}

export function buildVideoSamplingPlan(
  metadata: VideoSamplingMetadata,
  options: VideoSamplingOptions = {},
): VideoSamplingPlan {
  if (!Number.isFinite(metadata.duration) || metadata.duration <= 0) {
    throw new Error("the selected video does not expose a usable duration");
  }
  if (
    !Number.isFinite(metadata.videoWidth) ||
    !Number.isFinite(metadata.videoHeight) ||
    metadata.videoWidth <= 0 ||
    metadata.videoHeight <= 0
  ) {
    throw new Error("the selected video does not expose usable dimensions");
  }

  const framesPerSecond = Math.min(8, positiveFinite(options.framesPerSecond, 1.25));
  const automaticCap = Math.max(
    DEFAULT_MAX_FRAMES,
    Math.ceil(metadata.duration / MAX_AUTO_SAMPLE_SPACING_SECONDS),
  );
  const maxFrames = Math.min(
    MAX_SAMPLED_FRAMES,
    Math.max(4, Math.floor(positiveFinite(options.maxFrames, automaticCap))),
  );
  const targetCount = Math.max(4, Math.ceil(metadata.duration * framesPerSecond));
  const sampleCount = Math.min(maxFrames, targetCount);
  const analysisWidth = Math.min(360, Math.floor(metadata.videoWidth));
  const analysisHeight = Math.max(
    1,
    Math.round((analysisWidth / metadata.videoWidth) * metadata.videoHeight),
  );
  const times = Array.from({ length: sampleCount }, (_, index) => {
    const fraction = sampleCount === 1 ? 0.5 : 0.05 + (0.9 * index) / (sampleCount - 1);
    return Math.min(metadata.duration - 0.001, Math.max(0, metadata.duration * fraction));
  });

  return {
    framesPerSecond,
    maxFrames,
    sampleCount,
    analysisWidth,
    analysisHeight,
    times,
  };
}
