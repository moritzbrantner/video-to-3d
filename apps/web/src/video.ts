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

/** Display metadata of the sampled media (orientation already applied by the browser). */
export type SampledVideoInfo = {
  duration: number;
  displayWidth: number;
  displayHeight: number;
};

type FrameCallbackVideo = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: number, metadata: { mediaTime: number }) => void,
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

/**
 * Resolve with the media timestamp of the next frame the decoder presents, or
 * `null` when the browser lacks requestVideoFrameCallback or presents nothing
 * within the timeout. Register before seeking so the seeked frame is caught.
 */
function nextPresentedMediaTime(video: FrameCallbackVideo): Promise<number | null> {
  if (!video.requestVideoFrameCallback) return Promise.resolve(null);
  return new Promise((resolve) => {
    let handle = 0;
    const timer = setTimeout(() => {
      video.cancelVideoFrameCallback?.(handle);
      resolve(null);
    }, 500);
    handle = video.requestVideoFrameCallback!((_, metadata) => {
      clearTimeout(timer);
      resolve(metadata.mediaTime);
    });
  });
}

export async function sampleVideo(
  file: File,
  options: VideoSamplingOptions = {},
): Promise<SampledFrame[]> {
  return (await sampleVideoWithInfo(file, options)).frames;
}

export async function sampleVideoWithInfo(
  file: File,
  options: VideoSamplingOptions = {},
): Promise<{ frames: SampledFrame[]; info: SampledVideoInfo }> {
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
      let decodedTime: number | null = null;
      if (Math.abs(video.currentTime - time) > 0.001) {
        const presented = nextPresentedMediaTime(video as FrameCallbackVideo);
        video.currentTime = time;
        await waitFor(video, "seeked");
        decodedTime = await presented;
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
        // The decoded frame's media timestamp when the browser reports it;
        // otherwise only the seek position is known.
        presentedTime: decodedTime ?? video.currentTime,
        presentedTimeDecoded: decodedTime !== null,
      });
    }
    return {
      frames,
      info: {
        duration: video.duration,
        displayWidth: video.videoWidth,
        displayHeight: video.videoHeight,
      },
    };
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}
