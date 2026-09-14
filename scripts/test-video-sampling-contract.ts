import assert from "node:assert/strict";
import { buildVideoSamplingPlan } from "../apps/web/src/videoSampling";

const defaults = buildVideoSamplingPlan({
  duration: 10,
  videoWidth: 1920,
  videoHeight: 1080,
});
assert.equal(defaults.framesPerSecond, 1.25);
assert.equal(defaults.maxFrames, 18);
assert.equal(defaults.sampleCount, 13);
assert.equal(defaults.analysisWidth, 360);
assert.equal(defaults.analysisHeight, 203);
assert.equal(defaults.times[0], 0.5);
assert.equal(defaults.times.at(-1), 9.5);

const short = buildVideoSamplingPlan({
  duration: 1,
  videoWidth: 640,
  videoHeight: 480,
});
assert.equal(short.sampleCount, 4);
assert.deepEqual(
  short.times.map((time) => Number(time.toFixed(2))),
  [0.05, 0.35, 0.65, 0.95],
);

const capped = buildVideoSamplingPlan(
  { duration: 100, videoWidth: 3840, videoHeight: 2160 },
  { framesPerSecond: 99, maxFrames: 999 },
);
assert.equal(capped.framesPerSecond, 8);
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

console.log("video-sampling-contract=ok");
