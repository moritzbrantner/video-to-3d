export type Point2 = {
  x: number;
  y: number;
};

export type DenseReferencePatchRanges = {
  reference_frame: number;
  primary_start: number;
  primary_points: number;
  completion_start: number;
  completed_points: number;
};

const UNASSIGNED_REFERENCE = -1;
const AMBIGUOUS_REFERENCE = -2;

export function buildDensePointReferenceFrames(
  pointCount: number,
  patches: DenseReferencePatchRanges[],
): Int32Array {
  const references = new Int32Array(pointCount);
  references.fill(UNASSIGNED_REFERENCE);

  const assignRange = (start: number, count: number, referenceFrame: number) => {
    if (
      !Number.isSafeInteger(start) ||
      !Number.isSafeInteger(count) ||
      !Number.isSafeInteger(referenceFrame) ||
      start < 0 ||
      count < 0 ||
      referenceFrame < 0
    ) {
      return;
    }
    const end = Math.min(pointCount, start + count);
    for (let index = start; index < end; index += 1) {
      const current = references[index];
      if (current === UNASSIGNED_REFERENCE) {
        references[index] = referenceFrame;
      } else if (current !== referenceFrame) {
        references[index] = AMBIGUOUS_REFERENCE;
      }
    }
  };

  for (const patch of patches) {
    assignRange(patch.primary_start, patch.primary_points, patch.reference_frame);
    assignRange(patch.completion_start, patch.completed_points, patch.reference_frame);
  }
  return references;
}

export function triangleTextureReference(
  a: number,
  b: number,
  c: number,
  references: Int32Array,
): number | null {
  if (
    !Number.isSafeInteger(a) ||
    !Number.isSafeInteger(b) ||
    !Number.isSafeInteger(c) ||
    a < 0 ||
    b < 0 ||
    c < 0 ||
    a >= references.length ||
    b >= references.length ||
    c >= references.length
  ) {
    return null;
  }
  const reference = references[a];
  if (reference < 0 || references[b] !== reference || references[c] !== reference) {
    return null;
  }
  return reference;
}

export function affineTriangleTransform(
  source: [Point2, Point2, Point2],
  destination: [Point2, Point2, Point2],
): [number, number, number, number, number, number] | null {
  const [s0, s1, s2] = source;
  const [d0, d1, d2] = destination;
  const denominator =
    s0.x * (s1.y - s2.y) +
    s1.x * (s2.y - s0.y) +
    s2.x * (s0.y - s1.y);
  if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-9) return null;

  const a =
    (d0.x * (s1.y - s2.y) +
      d1.x * (s2.y - s0.y) +
      d2.x * (s0.y - s1.y)) /
    denominator;
  const c =
    (d0.x * (s2.x - s1.x) +
      d1.x * (s0.x - s2.x) +
      d2.x * (s1.x - s0.x)) /
    denominator;
  const e =
    (d0.x * (s1.x * s2.y - s2.x * s1.y) +
      d1.x * (s2.x * s0.y - s0.x * s2.y) +
      d2.x * (s0.x * s1.y - s1.x * s0.y)) /
    denominator;
  const b =
    (d0.y * (s1.y - s2.y) +
      d1.y * (s2.y - s0.y) +
      d2.y * (s0.y - s1.y)) /
    denominator;
  const d =
    (d0.y * (s2.x - s1.x) +
      d1.y * (s0.x - s2.x) +
      d2.y * (s1.x - s0.x)) /
    denominator;
  const f =
    (d0.y * (s1.x * s2.y - s2.x * s1.y) +
      d1.y * (s2.x * s0.y - s0.x * s2.y) +
      d2.y * (s0.x * s1.y - s1.x * s0.y)) /
    denominator;
  const transform = [a, b, c, d, e, f] as const;
  return transform.every(Number.isFinite) ? [...transform] : null;
}
