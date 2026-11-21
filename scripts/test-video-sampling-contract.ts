import assert from "node:assert/strict";
import { buildVideoSamplingPlan } from "../apps/web/src/videoSampling";
import { ppmSize, samplingReport } from "../tools/check_reference_sampling";

// Atlas 3.13.1: short clips at movie frame rates are sampled at 12 fps up to 120 frames.
const defaults = buildVideoSamplingPlan({
  duration: 3,
  videoWidth: 1920,
  videoHeight: 1080,
});
assert.equal(defaults.framesPerSecond, 12);
assert.equal(defaults.maxFrames, 120);
assert.equal(defaults.sampleCount, 36);
assert.equal(defaults.analysisWidth, 360);
assert.equal(defaults.analysisHeight, 203);
assert.equal(Number(defaults.times[0].toFixed(3)), 0.15);
assert.equal(Number(defaults.times.at(-1)!.toFixed(3)), 2.85);
assert.equal(buildVideoSamplingPlan({ duration: 10, videoWidth: 640, videoHeight: 480 }).sampleCount, 120);

// Long clips spread the 120-frame bound evenly; keyframe selection drops redundant samples.
const long = buildVideoSamplingPlan({ duration: 86.166667, videoWidth: 640, videoHeight: 480 });
assert.equal(long.maxFrames, 120);
assert.equal(long.sampleCount, 120);
assert.ok(long.times[1] - long.times[0] <= 1.0);
const explicit = buildVideoSamplingPlan(
  { duration: 86.166667, videoWidth: 640, videoHeight: 480 },
  { maxFrames: 18 },
);
assert.equal(explicit.sampleCount, 18, "an explicit cap still wins");
// Pinned fixture plans keep their recorded rate (Atlas 3.13.1).
assert.equal(
  buildVideoSamplingPlan({ duration: 10, videoWidth: 640, videoHeight: 480 }, { framesPerSecond: 1.25, maxFrames: 18 }).sampleCount,
  13,
);

const native = buildVideoSamplingPlan(
  { duration: 86.166667, videoWidth: 640, videoHeight: 480 },
  { profile: "local-reconstruction" },
);
assert.equal(native.sampleCount, 120);
assert.deepEqual([native.analysisWidth, native.analysisHeight], [640, 480]);
for (const [videoWidth, videoHeight] of [[4000, 4000], [1080, 1920], [3840, 2160]]) {
  const plan = buildVideoSamplingPlan(
    { duration: 200, videoWidth, videoHeight },
    { profile: "local-reconstruction" },
  );
  assert.ok(plan.analysisWidth * plan.analysisHeight * plan.sampleCount * 4 < 160 * 1024 * 1024);
}

const short = buildVideoSamplingPlan(
  { duration: 1, videoWidth: 640, videoHeight: 480 },
  { framesPerSecond: 1.25 },
);
assert.equal(short.sampleCount, 4);
assert.deepEqual(
  short.times.map((time) => Number(time.toFixed(2))),
  [0.05, 0.35, 0.65, 0.95],
);

const capped = buildVideoSamplingPlan(
  { duration: 100, videoWidth: 3840, videoHeight: 2160 },
  { framesPerSecond: 99, maxFrames: 999 },
);
assert.equal(capped.framesPerSecond, 15);
assert.equal(capped.maxFrames, 120);
assert.equal(capped.sampleCount, 120);

const portrait = buildVideoSamplingPlan({
  duration: 5,
  videoWidth: 240,
  videoHeight: 480,
});
assert.equal(portrait.analysisWidth, 240);
assert.equal(portrait.analysisHeight, 480);

assert.throws(
  () => buildVideoSamplingPlan({ duration: 0, videoWidth: 640, videoHeight: 480 }),
  /usable duration/,
);
assert.throws(
  () => buildVideoSamplingPlan({ duration: 1, videoWidth: 0, videoHeight: 480 }),
  /usable dimensions/,
);

// The reference-fixture sampling stage (#129) judges exported frames by the default plan:
// the fixtures declare a 1.5 s clip, which the 12 fps default samples 18 times.
const fixturePlan = buildVideoSamplingPlan({ duration: 1.5, videoWidth: 1920, videoHeight: 1080 });
assert.equal(fixturePlan.sampleCount, 18);
const fixtureTruth = {
  sampling: {
    source_width: 1920,
    source_height: 1080,
    duration_seconds: 1.5,
    times: fixturePlan.times,
    analysis_width: 360,
    analysis_height: 203,
  },
  frames: [],
};
const fixtureSizes = fixturePlan.times.map((): [number, number] => [360, 203]);
assert.equal(samplingReport(fixtureTruth, fixtureSizes).metrics.mismatched_frames, 0);
assert.equal(samplingReport(fixtureTruth, fixtureSizes).metrics.max_time_error_seconds, 0);
assert.equal(
  samplingReport(fixtureTruth, [...fixtureSizes.slice(1), [480, 270]]).metrics.mismatched_frames,
  1,
);
assert.equal(samplingReport(fixtureTruth, fixtureSizes.slice(1)).metrics.mismatched_frames, 1);
const shifted = {
  ...fixtureTruth,
  sampling: { ...fixtureTruth.sampling, times: fixturePlan.times.map((time) => time + 0.01) },
};
assert.ok(samplingReport(shifted, fixtureSizes).metrics.max_time_error_seconds > 0.009);
assert.deepEqual(ppmSize(new TextEncoder().encode("P6\n360 203\n255\n")), [360, 203]);

console.log("video-sampling-contract=ok");
