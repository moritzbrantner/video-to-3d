import type { SampledFrame } from "./reconstruction";
import type { SampledVideoInfo } from "./video";

export type ReadinessVerdict = "ready" | "marginal" | "unsuitable";

export type ReadinessIssue = {
  code: string;
  severity: "warning" | "blocking";
  frames: number[];
  message: string;
};

export type ReadinessPair = {
  from_frame: number;
  to_frame: number;
  matches: number;
  overlap: number;
  median_motion_pixels: number;
  residual_p75_pixels: number;
  residuals_epipolar_coherent: boolean;
  secondary_motion_fraction: number;
  motion: "duplicate" | "static" | "rotation_or_planar" | "parallax" | "unknown";
};

export type ReadinessReport = {
  schema_version: number;
  geometric_verdict: ReadinessVerdict;
  generative_paths_allowed: boolean;
  issues: ReadinessIssue[];
  pairs: ReadinessPair[];
  sampling: {
    median_interval_seconds: number;
    max_seek_error_seconds: number;
    decoded_time_samples: number;
  };
};

type ReadinessWasm = {
  default: () => Promise<unknown>;
  assess_input_readiness: (request: unknown) => unknown;
};

let wasmPromise: Promise<ReadinessWasm> | null = null;

function loadWasm(): Promise<ReadinessWasm> {
  if (!wasmPromise) {
    const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    const importModule = new Function("url", "return import(url)") as (
      url: string,
    ) => Promise<ReadinessWasm>;
    wasmPromise = importModule(`${basePath}/wasm/video_to_3d_wasm.js`).then(async (module) => {
      await module.default();
      return module;
    });
  }
  return wasmPromise;
}

function plain(value: unknown): unknown {
  if (value instanceof Map) {
    return Object.fromEntries([...value].map(([key, inner]) => [key, plain(inner)]));
  }
  if (Array.isArray(value)) return value.map(plain);
  return value;
}

/** Rust-owned input readiness evidence for the sampled frames. */
export async function assessReadiness(
  frames: SampledFrame[],
  info: SampledVideoInfo,
): Promise<ReadinessReport> {
  const wasm = await loadWasm();
  return plain(
    wasm.assess_input_readiness({
      frames: frames.map(({ width, height, rgba }) => ({ width, height, rgba })),
      metadata: {
        duration_seconds: info.duration,
        display_width: info.displayWidth,
        display_height: info.displayHeight,
        rotation_degrees: 0,
        requested_times: frames.map((frame) => frame.time),
        presented_times: frames.map((frame) => frame.presentedTime),
        presented_time_sources: frames.map((frame) =>
          frame.presentedTimeDecoded ? "decoded_frame" : "seek_position",
        ),
      },
    }),
  ) as ReadinessReport;
}
