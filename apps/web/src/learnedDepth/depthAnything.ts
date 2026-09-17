import type { SampledFrame } from "../reconstruction";
import { learnedProviderDescriptor } from "./catalog";
import { resampleRelativeDepth } from "./resample";
import type { LearnedProviderSession, RelativeDepthFrameEvidence } from "./types";

const MODEL_ID = "onnx-community/depth-anything-v2-small";

type DepthTensor = {
  data: Float32Array | ArrayLike<number>;
  dims: number[];
};

type DepthPipelineResult = {
  predicted_depth: DepthTensor;
};

type DepthPipeline = ((input: unknown) => Promise<DepthPipelineResult>) & {
  dispose?: () => Promise<void> | void;
};

type TransformersRuntime = {
  RawImage: new (data: Uint8Array, width: number, height: number, channels: 4) => unknown;
  pipeline: (
    task: "depth-estimation",
    model: string,
    options?: Record<string, unknown>,
  ) => Promise<DepthPipeline>;
};

function webGpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export async function createDepthAnythingSession(): Promise<LearnedProviderSession> {
  const runtime = (await import("@huggingface/transformers")) as unknown as TransformersRuntime;
  const useWebGpu = webGpuAvailable();
  const pipeline = await runtime.pipeline("depth-estimation", MODEL_ID, {
    ...(useWebGpu ? { device: "webgpu", dtype: "q4f16" } : { dtype: "q8" }),
  });

  return {
    descriptor: learnedProviderDescriptor("depth-anything-v2-small"),
    backend: useWebGpu ? "webgpu" : "wasm",
    runtimeLabel: useWebGpu ? "Transformers.js · WebGPU · q4f16" : "Transformers.js · WASM · q8",
    async infer(frame: SampledFrame, frameIndex: number): Promise<RelativeDepthFrameEvidence> {
      const rawImage = new runtime.RawImage(frame.rgba, frame.width, frame.height, 4);
      const startedAt = performance.now();
      const output = await pipeline(rawImage);
      const inferenceMs = performance.now() - startedAt;
      const dims = output.predicted_depth.dims;
      const sourceHeight = dims.at(-2) ?? frame.height;
      const sourceWidth = dims.at(-1) ?? frame.width;
      const depth = resampleRelativeDepth(
        output.predicted_depth.data,
        sourceWidth,
        sourceHeight,
        frame.width,
        frame.height,
      );
      return {
        providerId: "depth-anything-v2-small",
        frameIndex,
        representation: "relative_depth",
        width: frame.width,
        height: frame.height,
        depth,
        inferenceMs,
      };
    },
    async dispose() {
      await pipeline.dispose?.();
    },
  };
}
