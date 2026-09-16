import {
  affineTriangleTransform,
  buildDensePointReferenceFrames,
  triangleTextureReference,
  type Point2,
} from "../apps/web/src/meshTexture";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function approximatelyEqual(left: number, right: number, epsilon = 1e-6): boolean {
  return Math.abs(left - right) <= epsilon;
}

function applyTransform(
  transform: [number, number, number, number, number, number],
  point: Point2,
): Point2 {
  const [a, b, c, d, e, f] = transform;
  return {
    x: a * point.x + c * point.y + e,
    y: b * point.x + d * point.y + f,
  };
}

const references = buildDensePointReferenceFrames(8, [
  {
    reference_frame: 2,
    primary_start: 0,
    primary_points: 3,
    completion_start: 6,
    completed_points: 1,
  },
  {
    reference_frame: 5,
    primary_start: 3,
    primary_points: 3,
    completion_start: 7,
    completed_points: 1,
  },
]);
assert(references[0] === 2 && references[2] === 2 && references[6] === 2, "first reference ranges were not preserved");
assert(references[3] === 5 && references[5] === 5 && references[7] === 5, "second reference ranges were not preserved");
assert(triangleTextureReference(0, 1, 2, references) === 2, "single-reference triangle should be texture eligible");
assert(triangleTextureReference(1, 2, 3, references) === null, "mixed-reference triangle must fail closed");

const ambiguous = buildDensePointReferenceFrames(3, [
  { reference_frame: 1, primary_start: 0, primary_points: 2, completion_start: 2, completed_points: 0 },
  { reference_frame: 4, primary_start: 1, primary_points: 2, completion_start: 3, completed_points: 0 },
]);
assert(ambiguous[1] < 0, "overlapping references must become ambiguous");
assert(triangleTextureReference(0, 1, 2, ambiguous) === null, "ambiguous range must never select a texture");

const source: [Point2, Point2, Point2] = [
  { x: 2, y: 3 },
  { x: 8, y: 4 },
  { x: 4, y: 11 },
];
const destination: [Point2, Point2, Point2] = [
  { x: 20, y: 30 },
  { x: 70, y: 35 },
  { x: 32, y: 90 },
];
const transform = affineTriangleTransform(source, destination);
assert(transform !== null, "non-degenerate source triangle should produce an affine transform");
source.forEach((point, index) => {
  const mapped = applyTransform(transform, point);
  assert(approximatelyEqual(mapped.x, destination[index].x), `affine transform x mismatch at vertex ${index}`);
  assert(approximatelyEqual(mapped.y, destination[index].y), `affine transform y mismatch at vertex ${index}`);
});
assert(
  affineTriangleTransform(
    [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }],
    destination,
  ) === null,
  "degenerate texture triangle must fail closed",
);

console.log("texture projection contract passed");
