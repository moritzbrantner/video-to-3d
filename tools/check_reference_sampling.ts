// Sampling stage of the reference fixtures (#129): the frames a fixture exports must be
// exactly what the browser would sample from its notional source clip. The plan comes
// from the browser's own `buildVideoSamplingPlan`, not from a restatement of it.
//
// Usage: bun tools/check_reference_sampling.ts <fixture-dir>
// Writes <fixture-dir>/sampling-report.json and prints it.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildVideoSamplingPlan } from "../apps/web/src/videoSampling";

type Truth = {
  sampling: {
    source_width: number;
    source_height: number;
    duration_seconds: number;
    times: number[];
    analysis_width: number;
    analysis_height: number;
  };
  frames: { image: string }[];
};

/** Width and height from a binary PPM (P6) header. */
export function ppmSize(bytes: Uint8Array): [number, number] {
  const header = new TextDecoder("ascii").decode(bytes.subarray(0, 64));
  const tokens = header
    .split("\n")
    .map((line) => line.replace(/#.*/, ""))
    .join(" ")
    .trim()
    .split(/\s+/);
  if (tokens[0] !== "P6") {
    throw new Error("not a binary PPM image");
  }
  return [Number(tokens[1]), Number(tokens[2])];
}

export function samplingReport(truth: Truth, imageSizes: [number, number][]) {
  const sampling = truth.sampling;
  // The browser's defaults, not the fixture's restated rate and cap: a fixture that
  // drifts from what the browser samples must fail here.
  const plan = buildVideoSamplingPlan({
    duration: sampling.duration_seconds,
    videoWidth: sampling.source_width,
    videoHeight: sampling.source_height,
  });
  const count = Math.max(plan.sampleCount, imageSizes.length, sampling.times.length);
  let mismatchedFrames = 0;
  let maxTimeError = 0;
  for (let index = 0; index < count; index += 1) {
    const size = imageSizes[index];
    const time = sampling.times[index];
    const planned = plan.times[index];
    if (
      size === undefined ||
      time === undefined ||
      planned === undefined ||
      size[0] !== plan.analysisWidth ||
      size[1] !== plan.analysisHeight
    ) {
      mismatchedFrames += 1;
      continue;
    }
    maxTimeError = Math.max(maxTimeError, Math.abs(time - planned));
  }
  return {
    available: true,
    contract: "apps/web/src/videoSampling.ts buildVideoSamplingPlan",
    metrics: {
      sample_count: imageSizes.length,
      expected_sample_count: plan.sampleCount,
      analysis_size: `${plan.analysisWidth}x${plan.analysisHeight}`,
      max_time_error_seconds: maxTimeError,
      mismatched_frames: mismatchedFrames,
    },
  };
}

if (import.meta.main) {
  const fixtureDir = process.argv[2];
  if (!fixtureDir) {
    throw new Error("usage: bun tools/check_reference_sampling.ts <fixture-dir>");
  }
  const truth: Truth = JSON.parse(readFileSync(join(fixtureDir, "truth", "truth.json"), "utf8"));
  const sizes = truth.frames.map((frame) => ppmSize(readFileSync(join(fixtureDir, frame.image))));
  const report = samplingReport(truth, sizes);
  const text = `${JSON.stringify(report, null, 2)}\n`;
  writeFileSync(join(fixtureDir, "sampling-report.json"), text);
  process.stdout.write(text);
}
