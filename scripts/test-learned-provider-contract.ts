import assert from "node:assert/strict";
import {
  LEARNED_PROVIDER_CATALOG,
  learnedProvidersForMode,
} from "../apps/web/src/learnedDepth/catalog";
import { median, percentile } from "../apps/web/src/learnedDepth/metrics";
import {
  normalizedFocalForLetterbox,
  recoverMogeDepth,
  solveMogeDepthShift,
  type MogeLetterbox,
} from "../apps/web/src/learnedDepth/mogeGeometry";

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

{
  const size = 64;
  const letterbox: MogeLetterbox = {
    sourceWidth: size,
    sourceHeight: size,
    targetSize: size,
    drawWidth: size,
    drawHeight: size,
    offsetX: 0,
    offsetY: 0,
    scale: 1,
  };
  const normalizedFocal = 0.9;
  const focalPixels = (normalizedFocal * Math.hypot(size, size)) / 2;
  assert.ok(Math.abs(normalizedFocalForLetterbox(focalPixels, letterbox) - normalizedFocal) < 1e-7);

  const trueShift = 0.45;
  const points = new Float32Array(size * size * 3);
  const confidence = new Float32Array(size * size).fill(1);
  const diagonal = Math.hypot(size, size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = y * size + x;
      const u = (2 * (x - (size - 1) * 0.5)) / diagonal;
      const v = (2 * (y - (size - 1) * 0.5)) / diagonal;
      const trueDepth = 1.5 + ((x * 3 + y * 5) % 17) * 0.03;
      points[index * 3] = (u / normalizedFocal) * trueDepth;
      points[index * 3 + 1] = (v / normalizedFocal) * trueDepth;
      points[index * 3 + 2] = trueDepth - trueShift;
    }
  }

  const solved = solveMogeDepthShift(points, confidence, size, letterbox, normalizedFocal);
  assert.ok(solved.validInputSamples >= 24);
  assert.ok(Math.abs(solved.shift - trueShift) < 1e-4, `${solved.shift} != ${trueShift}`);

  const recovered = recoverMogeDepth(points, confidence, 2, letterbox, focalPixels);
  assert.ok(Math.abs(recovered.shift - trueShift) < 1e-4);
  const center = Math.floor(size / 2) * size + Math.floor(size / 2);
  const expectedCenterDepth = (points[center * 3 + 2] + trueShift) * 2;
  assert.ok(Math.abs(recovered.depth[center] - expectedCenterDepth) < 1e-4);
}

console.log("learned provider contract: ok");
