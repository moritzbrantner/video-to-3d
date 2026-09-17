import assert from "node:assert/strict";
import {
  LEARNED_PROVIDER_CATALOG,
  learnedProvidersForMode,
} from "../apps/web/src/learnedDepth/catalog";
import { median, percentile } from "../apps/web/src/learnedDepth/metrics";

assert.equal(
  new Set(LEARNED_PROVIDER_CATALOG.map(({ id }) => id)).size,
  LEARNED_PROVIDER_CATALOG.length,
);
assert.deepEqual(learnedProvidersForMode("classic"), []);
assert.deepEqual(learnedProvidersForMode("benchmark"), [
  "depth-anything-v2-small",
  "moge-2-vits",
]);
assert.equal(median([5, 1, 3]), 3);
assert.equal(median([4, 2]), 3);
assert.equal(percentile([10, 20, 30, 40, 50], 0.9), 50);
assert.equal(percentile([], 0.9), null);

for (const provider of LEARNED_PROVIDER_CATALOG) {
  assert.ok(provider.approximateDownloadBytes > 0);
  assert.equal(provider.preferredBackend, "webgpu");
  assert.ok(provider.modelReference.length > 0);
}

console.log("learned provider contract: ok");
