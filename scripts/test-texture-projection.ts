import { objectCameraView, sceneCameraView } from "../apps/web/src/sourceView";
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

// Orbit a glTF preview from a known CV camera center, with the original vertical FOV.
const camera = { frame_index: 0, rotation: [1,0,0, 0,1,0, 0,0,1], translation: [-1,-2,-3],
  authority: "registered_geometry" as const, confidence: 1, median_reprojection_error_pixels: 0 };
const sourceScene = sceneCameraView({ accepted_camera_evidence: [camera], calibrated_pair: { focal_pixels: 100 },
  points: [{x:1,y:2,z:7}] }, [{height:200}], null);
assert(sourceScene?.target === "1m -2m -7m", "viewer target must map accepted source camera into glTF");
assert(sourceScene.orbit === `0rad ${Math.PI / 2}rad 4m` && sourceScene.fov === `${Math.PI / 2}rad`,
  "source view should reproduce distance, rotation and focal");
const cropView = objectCameraView(camera, [140, 90, 160, 110], {width:200,height:200}, 100);
assert(approximatelyEqual(parseFloat(cropView.orbit), -Math.atan(0.5)), "off-center object should be viewed from its source ray");

console.log("texture projection contract passed");
