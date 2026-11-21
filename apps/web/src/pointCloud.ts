import type { ReconstructionResult, RegionConfidence } from "./reconstruction";

type PointCloudSource = Pick<
  ReconstructionResult,
  "calibrated_pair" | "points" | "dense_points"
> & { geometry_confidence: { regions: Pick<RegionConfidence, "origin" | "points">[] } };

export function observedPointCount(source: PointCloudSource): number {
  if (!source.calibrated_pair) {
    return 0;
  }
  return source.geometry_confidence.regions.reduce(
    (count, region) => count + (region.origin === "geometric_multi_view" ? region.points.count : 0),
    source.points.length,
  );
}

/** Snapshot only triangulated landmarks and original geometric multi-view depth samples. */
export function observedPointCloudPly(source: PointCloudSource): Blob {
  const count = observedPointCount(source);
  if (count === 0) {
    throw new Error("No calibrated, observed points are available to export.");
  }
  const header = [
    "ply",
    "format binary_little_endian 1.0",
    "comment video-to-3d observed geometry in arbitrary monocular units",
    "comment sample_kind 0 = triangulated sparse landmark, 1 = geometric multi-view depth",
    `element vertex ${count}`,
    "property float x", "property float y", "property float z",
    "property uchar red", "property uchar green", "property uchar blue",
    "property float confidence", "property uchar sample_kind",
    "end_header\n",
  ].join("\n");
  const bytes = new ArrayBuffer(count * 20);
  const view = new DataView(bytes);
  let offset = 0;
  function writePoint(x: number, y: number, z: number, r: number, g: number, b: number, confidence: number, kind: 0 | 1) {
    view.setFloat32(offset, x, true);
    view.setFloat32(offset + 4, y, true);
    view.setFloat32(offset + 8, z, true);
    view.setUint8(offset + 12, r);
    view.setUint8(offset + 13, g);
    view.setUint8(offset + 14, b);
    view.setFloat32(offset + 15, confidence, true);
    view.setUint8(offset + 19, kind);
    offset += 20;
  }
  for (const point of source.points) {
    writePoint(point.x, point.y, point.z, point.r, point.g, point.b, point.confidence, 0);
  }
  const { values, rgb } = source.dense_points;
  for (const { origin, points } of source.geometry_confidence.regions) {
    if (origin !== "geometric_multi_view") {
      continue;
    }
    for (let i = points.start; i < points.start + points.count; i++) {
      writePoint(values[i * 4], values[i * 4 + 1], values[i * 4 + 2],
        rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], values[i * 4 + 3], 1);
    }
  }
  return new Blob([header, bytes], { type: "application/octet-stream" });
}
