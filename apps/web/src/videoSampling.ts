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
  const maxFrames = Math.min(
    120,
    Math.max(4, Math.floor(positiveFinite(options.maxFrames, 18))),
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
