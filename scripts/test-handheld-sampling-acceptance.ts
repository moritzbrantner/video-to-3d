import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildVideoSamplingPlan } from "../apps/web/src/videoSampling";

type FixtureExpectation = {
  cases: Array<{
    id: string;
    preparation: {
      trim: { start_seconds: number; end_seconds: number };
      plan: ReturnType<typeof buildVideoSamplingPlan>;
      inventory: {
        objects: Array<{ id: string; visible_samples: number[]; coverage: string }>;
      };
    };
  }>;
};
const expected: FixtureExpectation = JSON.parse(
  readFileSync("tools/tests/fixtures/approved-handheld-expectations.json", "utf8"),
);
for (const fixture of expected.cases) {
  const { trim, inventory } = fixture.preparation;
  const actual = buildVideoSamplingPlan({
    duration: trim.end_seconds - trim.start_seconds,
    videoWidth: 640,
    videoHeight: 480,
  }, { maxFrames: 18 });
  assert.deepEqual(actual, fixture.preparation.plan);
  assert.equal(actual.sampleCount, 18);
  assert.equal(actual.analysisWidth, 360);
  assert.equal(actual.analysisHeight, 270);
  for (const object of inventory.objects) {
    assert.ok(object.visible_samples.length > 0);
    assert.ok(object.visible_samples.every((index) => index >= 0 && index < actual.sampleCount));
  }
}
console.log("approved-handheld-sampling-acceptance=ok (plan only; browser decode unverified)");
