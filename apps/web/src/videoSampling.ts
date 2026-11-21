export type VideoSamplingOptions = {
  framesPerSecond?: number;
  maxFrames?: number;
  profile?: "browser" | "local-reconstruction";
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
 * Typical input is a short clip of a few seconds at movie frame rates (Atlas 3.13.1), so
 * clips are sampled densely up to a fixed frame bound; the Rust keyframe selection then
 * drops redundant samples by measured overlap.
 */
export const DEFAULT_SAMPLING_FPS = 12;
export const MAX_SAMPLING_FPS = 15;
export const MAX_SAMPLED_FRAMES = 120;

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

  const framesPerSecond = Math.min(MAX_SAMPLING_FPS, positiveFinite(options.framesPerSecond, DEFAULT_SAMPLING_FPS));
  const local = options.profile === "local-reconstruction";
  const maxFrames = Math.min(
    MAX_SAMPLED_FRAMES,
    Math.max(4, Math.floor(positiveFinite(options.maxFrames, MAX_SAMPLED_FRAMES))),
  );
  const targetCount = Math.max(4, Math.ceil(metadata.duration * framesPerSecond));
  const sampleCount = Math.min(maxFrames, targetCount);
  // Bound the local RGBA transfer for landscape, portrait and square videos.
  const scale = local ? Math.min(
    1, 640 / Math.max(metadata.videoWidth, metadata.videoHeight),
    Math.sqrt((640 * 480) / (metadata.videoWidth * metadata.videoHeight)),
  ) : 1;
  const analysisWidth = local ? Math.max(1, Math.floor(scale * metadata.videoWidth))
    : Math.min(360, Math.floor(metadata.videoWidth));
  const analysisHeight = Math.max(
    1,
    local ? Math.floor(scale * metadata.videoHeight) :
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
