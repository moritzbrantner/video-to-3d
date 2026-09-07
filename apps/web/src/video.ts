import type { SampledFrame } from "./reconstruction";

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

export async function sampleVideo(file: File): Promise<SampledFrame[]> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.src = url;

  try {
    await waitFor(video, "loadedmetadata");
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      throw new Error("the selected video does not expose a usable duration");
    }

    const sampleCount = Math.min(18, Math.max(4, Math.ceil(video.duration * 1.25)));
    const analysisWidth = Math.min(360, video.videoWidth);
    const analysisHeight = Math.max(
      1,
      Math.round((analysisWidth / video.videoWidth) * video.videoHeight),
    );
    const canvas = document.createElement("canvas");
    canvas.width = analysisWidth;
    canvas.height = analysisHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      throw new Error("2D canvas is unavailable in this browser");
    }

    const frames: SampledFrame[] = [];
    for (let index = 0; index < sampleCount; index += 1) {
      const fraction =
        sampleCount === 1 ? 0.5 : 0.05 + (0.9 * index) / (sampleCount - 1);
      const time = Math.min(
        video.duration - 0.001,
        Math.max(0, video.duration * fraction),
      );
      if (Math.abs(video.currentTime - time) > 0.001) {
        video.currentTime = time;
        await waitFor(video, "seeked");
      }
      context.drawImage(video, 0, 0, analysisWidth, analysisHeight);
      const image = context.getImageData(0, 0, analysisWidth, analysisHeight);
      frames.push({
        width: analysisWidth,
        height: analysisHeight,
        rgba: Array.from(image.data),
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
