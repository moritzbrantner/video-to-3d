import type { SampledFrame } from "./reconstruction";
import {
  buildVideoSamplingPlan,
  type VideoSamplingOptions,
} from "./videoSampling";

export type { VideoSamplingOptions } from "./videoSampling";

function waitFor(target: EventTarget, event: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const onEvent = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error(`video failed while waiting for ${event}`));
    };
    const cleanup = () => {
      target.removeEventListener(event, onEvent);
      target.removeEventListener("error", onError);
    };
    target.addEventListener(event, onEvent, { once: true });
    target.addEventListener("error", onError, { once: true });
  });
}

export async function sampleVideo(
  file: File,
  options: VideoSamplingOptions = {},
): Promise<SampledFrame[]> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.src = url;

  try {
    await waitFor(video, "loadedmetadata");
    const plan = buildVideoSamplingPlan(
      {
        duration: video.duration,
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight,
      },
      options,
    );
    const canvas = document.createElement("canvas");
    canvas.width = plan.analysisWidth;
    canvas.height = plan.analysisHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      throw new Error("2D canvas is unavailable in this browser");
    }

    const frames: SampledFrame[] = [];
    for (const time of plan.times) {
      if (Math.abs(video.currentTime - time) > 0.001) {
        video.currentTime = time;
        await waitFor(video, "seeked");
      }
      context.drawImage(video, 0, 0, plan.analysisWidth, plan.analysisHeight);
      const image = context.getImageData(0, 0, plan.analysisWidth, plan.analysisHeight);
      frames.push({
        width: plan.analysisWidth,
        height: plan.analysisHeight,
        rgba: new Uint8Array(
          image.data.buffer,
          image.data.byteOffset,
          image.data.byteLength,
        ),
        thumbnail: canvas.toDataURL("image/jpeg", 0.68),
        time,
      });
    }
    return frames;
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}
