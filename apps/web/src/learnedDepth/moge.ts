import type { SampledFrame } from "../reconstruction";
import { learnedProviderDescriptor } from "./catalog";
import type { MogeLetterbox } from "./mogeGeometry";
import type { AffinePointMapFrameEvidence, LearnedProviderSession } from "./types";

const SIZE = 448;
const PLANE = SIZE * SIZE;
const DEFAULT_MODEL_BASE =
  "https://huggingface.co/litert-community/MoGe-2-LiteRT/resolve/main/";

type LiteTensor = {
  data(): Promise<Float32Array | ArrayLike<number>>;
  delete(): void;
};

type LiteModel = {
  run(inputs: unknown[]): Promise<LiteTensor[]>;
  delete?: () => void;
};

type LiteRuntime = {
  Tensor: {
    fromTypedArray(data: Float32Array, shape: number[]): unknown;
  };
  isWebGPUSupported(): boolean;
  loadAndCompile(bytes: Uint8Array, options: { accelerator: "webgpu" | "wasm" }): Promise<LiteModel>;
  loadLiteRt(wasmDir: string, options: { threads: boolean }): Promise<unknown>;
};

function basePath(): string {
  return process.env.NEXT_PUBLIC_BASE_PATH ?? "";
}

function modelUrl(accelerator: "webgpu" | "wasm"): string {
  const modelBase = DEFAULT_MODEL_BASE.replace(/\/?$/, "/");
  return `${modelBase}${accelerator === "webgpu" ? "moge_fp16.tflite" : "moge.tflite"}`;
}

async function fetchModelBytes(url: string): Promise<Uint8Array> {
  const cache = typeof caches === "undefined" ? null : await caches.open("video-to-3d-moge-v1");
  const cached = cache ? await cache.match(url) : null;
  if (cached) return new Uint8Array(await cached.arrayBuffer());

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`MoGe model download failed with HTTP ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (cache) {
    await cache.put(url, new Response(bytes.slice().buffer));
  }
  return bytes;
}

function preprocess(frame: SampledFrame): { nchw: Float32Array; letterbox: MogeLetterbox } {
  const source = document.createElement("canvas");
  source.width = frame.width;
  source.height = frame.height;
  const sourceContext = source.getContext("2d");
  if (!sourceContext) throw new Error("MoGe preprocessing requires a 2D canvas context");
  const rgba = new Uint8ClampedArray(
    frame.rgba.buffer,
    frame.rgba.byteOffset,
    frame.rgba.byteLength,
  );
  sourceContext.putImageData(new ImageData(rgba, frame.width, frame.height), 0, 0);

  const target = document.createElement("canvas");
  target.width = SIZE;
  target.height = SIZE;
  const targetContext = target.getContext("2d", { willReadFrequently: true });
  if (!targetContext) throw new Error("MoGe preprocessing requires a readable 2D canvas context");

  const scale = Math.min(SIZE / frame.width, SIZE / frame.height);
  const drawWidth = Math.round(frame.width * scale);
  const drawHeight = Math.round(frame.height * scale);
  const offsetX = Math.floor((SIZE - drawWidth) / 2);
  const offsetY = Math.floor((SIZE - drawHeight) / 2);
  targetContext.fillStyle = "#7f7f7f";
  targetContext.fillRect(0, 0, SIZE, SIZE);
  targetContext.drawImage(source, offsetX, offsetY, drawWidth, drawHeight);

  const image = targetContext.getImageData(0, 0, SIZE, SIZE).data;
  const nchw = new Float32Array(3 * PLANE);
  for (let index = 0; index < PLANE; index += 1) {
    nchw[index] = image[index * 4] / 255;
    nchw[PLANE + index] = image[index * 4 + 1] / 255;
    nchw[PLANE * 2 + index] = image[index * 4 + 2] / 255;
  }
  return {
    nchw,
    letterbox: {
      sourceWidth: frame.width,
      sourceHeight: frame.height,
      targetSize: SIZE,
      drawWidth,
      drawHeight,
      offsetX,
      offsetY,
      scale,
    },
  };
}

function sampledAbsMax(values: Float32Array): number {
  let maximum = 0;
  const step = Math.max(1, Math.floor(values.length / 5000));
  for (let index = 0; index < values.length; index += step) {
    maximum = Math.max(maximum, Math.abs(values[index]));
  }
  return maximum;
}

function asFloat32(values: Float32Array | ArrayLike<number>): Float32Array {
  return values instanceof Float32Array ? values.slice() : Float32Array.from(values);
}

function resolveOutputs(buffers: Float32Array[]): {
  points: Float32Array;
  confidence: Float32Array;
  metricScale: number | null;
} {
  const vectors = buffers.filter((buffer) => buffer.length === PLANE * 3);
  const confidence = buffers.find((buffer) => buffer.length === PLANE);
  const metricScale = buffers.find((buffer) => buffer.length === 1);
  if (vectors.length !== 2 || !confidence || !metricScale) {
    throw new Error("MoGe output contract mismatch: expected points, normals, mask, and metric scale");
  }
  const points = sampledAbsMax(vectors[0]) > 2 ? vectors[0] : vectors[1];
  return {
    points,
    confidence,
    metricScale: Number.isFinite(metricScale[0]) ? metricScale[0] : null,
  };
}

export async function createMogeSession(): Promise<LearnedProviderSession> {
  const runtime = (await import("@litertjs/core")) as unknown as LiteRuntime;
  await runtime.loadLiteRt(`${location.origin}${basePath()}/litert-wasm/`, { threads: false });

  let accelerator: "webgpu" | "wasm" = runtime.isWebGPUSupported() ? "webgpu" : "wasm";
  let model: LiteModel;
  try {
    model = await runtime.loadAndCompile(await fetchModelBytes(modelUrl(accelerator)), {
      accelerator,
    });
  } catch (error) {
    if (accelerator === "wasm") throw error;
    accelerator = "wasm";
    model = await runtime.loadAndCompile(await fetchModelBytes(modelUrl(accelerator)), {
      accelerator,
    });
  }

  return {
    descriptor: learnedProviderDescriptor("moge-2-vits"),
    backend: accelerator,
    runtimeLabel: `LiteRT.js · ${accelerator === "webgpu" ? "WebGPU · fp16 weights" : "WASM · fp32"}`,
    async infer(frame: SampledFrame, frameIndex: number): Promise<AffinePointMapFrameEvidence> {
      const { nchw, letterbox } = preprocess(frame);
      const input = runtime.Tensor.fromTypedArray(nchw, [1, 3, SIZE, SIZE]) as {
        delete?: () => void;
      };
      const startedAt = performance.now();
      const outputs = await model.run([input]);
      const buffers: Float32Array[] = [];
      try {
        for (const output of outputs) buffers.push(asFloat32(await output.data()));
      } finally {
        for (const output of outputs) output.delete();
        input.delete?.();
      }
      const inferenceMs = performance.now() - startedAt;
      const { points, confidence, metricScale } = resolveOutputs(buffers);
      return {
        providerId: "moge-2-vits",
        frameIndex,
        representation: "affine_point_map",
        width: SIZE,
        height: SIZE,
        points,
        confidence,
        metricScale,
        letterbox,
        inferenceMs,
      };
    },
    dispose() {
      model.delete?.();
    },
  };
}
