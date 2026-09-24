import assert from "node:assert/strict";
import {
  DEFAULT_LEARNED_RECONSTRUCTION_MODE,
  LEARNED_PROVIDER_CATALOG,
  learnedModeFromQueryParam,
  learnedModeQueryParam,
  learnedProvidersForMode,
} from "../apps/web/src/learnedDepth/catalog";
import { median, percentile } from "../apps/web/src/learnedDepth/metrics";
import { resampleRelativeDepth } from "../apps/web/src/learnedDepth/resample";

assert.equal(
  new Set(LEARNED_PROVIDER_CATALOG.map(({ id }) => id)).size,
  LEARNED_PROVIDER_CATALOG.length,
);
assert.deepEqual(learnedProvidersForMode("classic"), []);
assert.deepEqual(learnedProvidersForMode("benchmark"), [
  "depth-anything-v2-small",
  "moge-2-vits",
]);
assert.equal(learnedModeFromQueryParam(null), DEFAULT_LEARNED_RECONSTRUCTION_MODE);
assert.equal(learnedModeFromQueryParam("unknown-model"), DEFAULT_LEARNED_RECONSTRUCTION_MODE);
assert.equal(learnedModeFromQueryParam("depth-anything-v2-small"), "depth-anything-v2-small");
assert.equal(learnedModeFromQueryParam("moge-2-vits"), "moge-2-vits");
assert.equal(learnedModeFromQueryParam("benchmark"), "benchmark");
assert.equal(learnedModeQueryParam("classic"), null);
assert.equal(learnedModeQueryParam("moge-2-vits"), "moge-2-vits");
assert.equal(median([5, 1, 3]), 3);
assert.equal(median([4, 2]), 3);
assert.equal(percentile([10, 20, 30, 40, 50], 0.9), 50);
assert.equal(percentile([], 0.9), null);

const resampledDepth = resampleRelativeDepth(
  new Float32Array([0, 10, 20, 30]),
  2,
  2,
  3,
  3,
);
assert.deepEqual(Array.from(resampledDepth), [0, 5, 10, 10, 15, 20, 20, 25, 30]);

for (const provider of LEARNED_PROVIDER_CATALOG) {
  assert.ok(provider.approximateDownloadBytes > 0);
  assert.equal(provider.preferredBackend, "webgpu");
  assert.ok(provider.modelReference.length > 0);
}

console.log("learned provider contract: ok");
