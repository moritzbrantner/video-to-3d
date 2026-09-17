export type MogeLetterbox = {
  sourceWidth: number;
  sourceHeight: number;
  targetSize: number;
  drawWidth: number;
  drawHeight: number;
  offsetX: number;
  offsetY: number;
  scale: number;
};

export type MogeDepthRecovery = {
  depth: Float32Array;
  width: number;
  height: number;
  shift: number;
  normalizedFocal: number;
  validInputSamples: number;
};

const MASK_THRESHOLD = 0.5;
const MIN_SHIFT_SAMPLES = 24;
const MAX_GAUSS_NEWTON_STEPS = 12;

function normalizedUv(pixel: number, size: number): number {
  const center = (size - 1) * 0.5;
  const diagonal = Math.sqrt(size * size + size * size);
  return (2 * (pixel - center)) / diagonal;
}

export function normalizedFocalForLetterbox(
  focalPixels: number,
  letterbox: MogeLetterbox,
): number {
  if (!Number.isFinite(focalPixels) || focalPixels <= 0) {
    throw new Error("MoGe focal recovery requires a positive accepted focal length");
  }
  return (2 * focalPixels * letterbox.scale) / Math.hypot(letterbox.targetSize, letterbox.targetSize);
}

function objective(
  points: Float32Array,
  confidence: Float32Array,
  size: number,
  letterbox: MogeLetterbox,
  focal: number,
  shift: number,
): number {
  let error = 0;
  let count = 0;
  const sampleStride = Math.max(1, Math.floor(size / 64));
  const right = letterbox.offsetX + letterbox.drawWidth;
  const bottom = letterbox.offsetY + letterbox.drawHeight;
  for (let y = letterbox.offsetY + 1; y < bottom - 1; y += sampleStride) {
    for (let x = letterbox.offsetX + 1; x < right - 1; x += sampleStride) {
      const pixel = y * size + x;
      if (!(confidence[pixel] > MASK_THRESHOLD)) continue;
      const offset = pixel * 3;
      const px = points[offset];
      const py = points[offset + 1];
      const pz = points[offset + 2];
      const denominator = pz + shift;
      if (
        !Number.isFinite(px) ||
        !Number.isFinite(py) ||
        !Number.isFinite(denominator) ||
        Math.abs(denominator) < 1e-6
      ) {
        continue;
      }
      const u = normalizedUv(x, size);
      const v = normalizedUv(y, size);
      const rx = (focal * px) / denominator - u;
      const ry = (focal * py) / denominator - v;
      error += rx * rx + ry * ry;
      count += 1;
    }
  }
  return count >= MIN_SHIFT_SAMPLES ? error / count : Number.POSITIVE_INFINITY;
}

export function solveMogeDepthShift(
  points: Float32Array,
  confidence: Float32Array,
  size: number,
  letterbox: MogeLetterbox,
  normalizedFocal: number,
): { shift: number; validInputSamples: number } {
  if (points.length !== size * size * 3 || confidence.length !== size * size) {
    throw new Error("MoGe shift recovery received inconsistent point or confidence dimensions");
  }
  if (!Number.isFinite(normalizedFocal) || normalizedFocal <= 0) {
    throw new Error("MoGe shift recovery requires a positive normalized focal length");
  }

  const sampleStride = Math.max(1, Math.floor(size / 64));
  const right = letterbox.offsetX + letterbox.drawWidth;
  const bottom = letterbox.offsetY + letterbox.drawHeight;
  let crossNumerator = 0;
  let crossDenominator = 0;
  let validInputSamples = 0;
  let zAbsSum = 0;

  for (let y = letterbox.offsetY + 1; y < bottom - 1; y += sampleStride) {
    for (let x = letterbox.offsetX + 1; x < right - 1; x += sampleStride) {
      const pixel = y * size + x;
      if (!(confidence[pixel] > MASK_THRESHOLD)) continue;
      const offset = pixel * 3;
      const px = points[offset];
      const py = points[offset + 1];
      const pz = points[offset + 2];
      if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
      const u = normalizedUv(x, size);
      const v = normalizedUv(y, size);
      crossNumerator +=
        u * (normalizedFocal * px - u * pz) +
        v * (normalizedFocal * py - v * pz);
      crossDenominator += u * u + v * v;
      zAbsSum += Math.abs(pz);
      validInputSamples += 1;
    }
  }

  if (validInputSamples < MIN_SHIFT_SAMPLES || crossDenominator <= 1e-9) {
    throw new Error(
      `MoGe shift recovery needs at least ${MIN_SHIFT_SAMPLES} confident non-degenerate samples`,
    );
  }

  const crossShift = crossNumerator / crossDenominator;
  let shift =
    objective(points, confidence, size, letterbox, normalizedFocal, crossShift) <=
    objective(points, confidence, size, letterbox, normalizedFocal, 0)
      ? crossShift
      : 0;
  const typicalZ = Math.max(zAbsSum / validInputSamples, 1e-3);

  // The reference implementation solves the same one-dimensional reprojection
  // objective with Levenberg-Marquardt. Damped Gauss-Newton is deterministic
  // here and uses the cross-multiplied least-squares solution as its seed.
  for (let iteration = 0; iteration < MAX_GAUSS_NEWTON_STEPS; iteration += 1) {
    let gradient = 0;
    let hessian = 0;
    let used = 0;
    for (let y = letterbox.offsetY + 1; y < bottom - 1; y += sampleStride) {
      for (let x = letterbox.offsetX + 1; x < right - 1; x += sampleStride) {
        const pixel = y * size + x;
        if (!(confidence[pixel] > MASK_THRESHOLD)) continue;
        const offset = pixel * 3;
        const px = points[offset];
        const py = points[offset + 1];
        const denominator = points[offset + 2] + shift;
        if (
          !Number.isFinite(px) ||
          !Number.isFinite(py) ||
          !Number.isFinite(denominator) ||
          Math.abs(denominator) < 1e-6
        ) {
          continue;
        }
        const u = normalizedUv(x, size);
        const v = normalizedUv(y, size);
        const inv = 1 / denominator;
        const rx = normalizedFocal * px * inv - u;
        const ry = normalizedFocal * py * inv - v;
        const jx = -normalizedFocal * px * inv * inv;
        const jy = -normalizedFocal * py * inv * inv;
        gradient += jx * rx + jy * ry;
        hessian += jx * jx + jy * jy;
        used += 1;
      }
    }
    if (used < MIN_SHIFT_SAMPLES || !Number.isFinite(hessian) || hessian <= 1e-12) break;
    const rawStep = -gradient / (hessian + 1e-6);
    const maxStep = typicalZ * 0.5;
    const step = Math.max(-maxStep, Math.min(maxStep, rawStep));
    if (!Number.isFinite(step)) break;

    const currentError = objective(points, confidence, size, letterbox, normalizedFocal, shift);
    let acceptedStep = step;
    let candidate = shift + acceptedStep;
    let candidateError = objective(
      points,
      confidence,
      size,
      letterbox,
      normalizedFocal,
      candidate,
    );
    for (let retry = 0; retry < 5 && candidateError > currentError; retry += 1) {
      acceptedStep *= 0.5;
      candidate = shift + acceptedStep;
      candidateError = objective(
        points,
        confidence,
        size,
        letterbox,
        normalizedFocal,
        candidate,
      );
    }
    if (!(candidateError <= currentError)) break;
    shift = candidate;
    if (Math.abs(acceptedStep) <= Math.max(1e-6, typicalZ * 1e-5)) break;
  }

  if (!Number.isFinite(shift)) throw new Error("MoGe shift recovery produced a non-finite shift");
  return { shift, validInputSamples };
}

function sampleBilinear(
  values: Float32Array,
  size: number,
  x: number,
  y: number,
): number {
  const x0 = Math.max(0, Math.min(size - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(size - 1, Math.floor(y)));
  const x1 = Math.min(size - 1, x0 + 1);
  const y1 = Math.min(size - 1, y0 + 1);
  const tx = Math.max(0, Math.min(1, x - x0));
  const ty = Math.max(0, Math.min(1, y - y0));
  const a = values[y0 * size + x0];
  const b = values[y0 * size + x1];
  const c = values[y1 * size + x0];
  const d = values[y1 * size + x1];
  if (![a, b, c, d].every(Number.isFinite)) return Number.NaN;
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

export function recoverMogeDepth(
  points: Float32Array,
  confidence: Float32Array,
  metricScale: number | null,
  letterbox: MogeLetterbox,
  focalPixels: number,
): MogeDepthRecovery {
  const size = letterbox.targetSize;
  const normalizedFocal = normalizedFocalForLetterbox(focalPixels, letterbox);
  const { shift, validInputSamples } = solveMogeDepthShift(
    points,
    confidence,
    size,
    letterbox,
    normalizedFocal,
  );
  const scale = metricScale !== null && Number.isFinite(metricScale) && metricScale > 0 ? metricScale : 1;
  const modelDepth = new Float32Array(size * size);
  for (let index = 0; index < modelDepth.length; index += 1) {
    const z = points[index * 3 + 2] + shift;
    modelDepth[index] = confidence[index] > MASK_THRESHOLD && Number.isFinite(z) && z > 1e-6
      ? z * scale
      : Number.NaN;
  }

  const depth = new Float32Array(letterbox.sourceWidth * letterbox.sourceHeight);
  for (let y = 0; y < letterbox.sourceHeight; y += 1) {
    const modelY =
      letterbox.offsetY + ((y + 0.5) * letterbox.drawHeight) / letterbox.sourceHeight - 0.5;
    for (let x = 0; x < letterbox.sourceWidth; x += 1) {
      const modelX =
        letterbox.offsetX + ((x + 0.5) * letterbox.drawWidth) / letterbox.sourceWidth - 0.5;
      depth[y * letterbox.sourceWidth + x] = sampleBilinear(modelDepth, size, modelX, modelY);
    }
  }

  return {
    depth,
    width: letterbox.sourceWidth,
    height: letterbox.sourceHeight,
    shift,
    normalizedFocal,
    validInputSamples,
  };
}
