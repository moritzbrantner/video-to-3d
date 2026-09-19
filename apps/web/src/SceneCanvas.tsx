"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReconstructionResult } from "./reconstruction";
import {
  affineTriangleTransform,
  buildDensePointReferenceFrames,
  triangleTextureReference,
  type Point2,
} from "./meshTexture";

type TextureFrame = {
  width: number;
  height: number;
  thumbnail: string;
};

const EMPTY_TEXTURE_FRAMES: TextureFrame[] = [];

type SceneCanvasProps = {
  reconstruction: ReconstructionResult;
  textureFrames?: TextureFrame[];
  selectedFrameIndex?: number | null;
  onSelectFrame?: (frameIndex: number) => void;
};

type ViewState = {
  yaw: number;
  pitch: number;
  zoom: number;
};

type RenderMode = "model" | "evidence";

type CameraHitTarget = {
  frameIndex: number;
  x: number;
  y: number;
  radius: number;
};

type DragState = {
  x: number;
  y: number;
  startX: number;
  startY: number;
  moved: boolean;
};

function clampedChannel(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

export function SceneCanvas({
  reconstruction,
  textureFrames = EMPTY_TEXTURE_FRAMES,
  selectedFrameIndex = null,
  onSelectFrame,
}: SceneCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const cameraHitTargetsRef = useRef<CameraHitTarget[]>([]);
  const [view, setView] = useState<ViewState>({ yaw: -0.45, pitch: 0.18, zoom: 1 });
  const cameraState = reconstruction.camera_state;
  const densePoints = reconstruction.dense_points;
  const denseGridSites = reconstruction.dense_grid_sites;
  const meshTriangles = reconstruction.mesh_triangles;
  const densePointReferenceFrames = useMemo(
    () =>
      buildDensePointReferenceFrames(
        densePoints.length,
        reconstruction.dense.reference_patches,
      ),
    [densePoints.length, reconstruction.dense.reference_patches],
  );
  const textureReferenceFrames = useMemo(
    () =>
      [
        ...new Set(
          reconstruction.dense.reference_patches.map((patch) => patch.reference_frame),
        ),
      ].filter((frameIndex) => textureFrames[frameIndex]),
    [reconstruction.dense.reference_patches, textureFrames],
  );
  const [textureImages, setTextureImages] = useState<
    Array<{ source: string; image: HTMLImageElement } | null>
  >([]);
  const [textureInteractionActive, setTextureInteractionActive] = useState(false);
  const textureInteractionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hasRegisteredGeometry = cameraState.calibrated_seed_cameras.length === 2;
  const acceptedCameras = useMemo(
    () => [...cameraState.calibrated_seed_cameras, ...cameraState.registered_cameras],
    [cameraState],
  );
  const displayCameras = useMemo(
    () => (hasRegisteredGeometry ? acceptedCameras : cameraState.approximate_motion_samples),
    [acceptedCameras, cameraState, hasRegisteredGeometry],
  );
  const hasSurfaceModel = meshTriangles.length > 0;
  const [renderMode, setRenderMode] = useState<RenderMode>(
    hasSurfaceModel ? "model" : "evidence",
  );

  useEffect(() => {
    setRenderMode(hasSurfaceModel ? "model" : "evidence");
    setView((current) => ({ ...current, zoom: 1 }));
  }, [hasSurfaceModel, reconstruction]);

  useEffect(() => {
    let cancelled = false;
    const loaded = Array<{ source: string; image: HTMLImageElement } | null>(
      textureFrames.length,
    ).fill(null);
    setTextureImages(loaded);
    const pending = textureReferenceFrames.flatMap((index) => {
      const frame = textureFrames[index];
      if (!frame) return [];
      const image = new Image();
      image.onload = () => {
        if (cancelled) return;
        loaded[index] = { source: frame.thumbnail, image };
        setTextureImages([...loaded]);
      };
      image.src = frame.thumbnail;
      return [image];
    });
    return () => {
      cancelled = true;
      for (const image of pending) image.onload = null;
    };
  }, [textureFrames, textureReferenceFrames]);

  useEffect(
    () => () => {
      if (textureInteractionTimeoutRef.current !== null) {
        clearTimeout(textureInteractionTimeoutRef.current);
      }
    },
    [],
  );

  const textureEligibleTriangles = useMemo(() => {
    if (denseGridSites.length !== densePoints.length) return 0;
    let accepted = 0;
    for (let triangleIndex = 0; triangleIndex < meshTriangles.length; triangleIndex += 1) {
      const base = triangleIndex * 3;
      const referenceFrame = triangleTextureReference(
        meshTriangles.indices[base],
        meshTriangles.indices[base + 1],
        meshTriangles.indices[base + 2],
        densePointReferenceFrames,
      );
      if (referenceFrame !== null && textureFrames[referenceFrame]) accepted += 1;
    }
    return accepted;
  }, [denseGridSites.length, densePointReferenceFrames, densePoints.length, meshTriangles, textureFrames]);

  const bounds = useMemo(() => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    let positionCount = 0;

    const includePosition = (x: number, y: number, z: number) => {
      min[0] = Math.min(min[0], x);
      min[1] = Math.min(min[1], y);
      min[2] = Math.min(min[2], z);
      max[0] = Math.max(max[0], x);
      max[1] = Math.max(max[1], y);
      max[2] = Math.max(max[2], z);
      positionCount += 1;
    };
    const includeDensePoint = (index: number) => {
      if (index < 0 || index >= densePoints.length) return;
      const base = index * 4;
      includePosition(
        densePoints.values[base],
        densePoints.values[base + 1],
        densePoints.values[base + 2],
      );
    };

    if (renderMode === "evidence") {
      for (let index = 0; index < densePoints.length; index += 1) {
        includeDensePoint(index);
      }
      for (const point of reconstruction.points) {
        includePosition(point.x, point.y, point.z);
      }
      for (const camera of displayCameras) {
        includePosition(camera.x, camera.y, camera.z);
      }
    } else if (meshTriangles.length > 0) {
      for (let triangleIndex = 0; triangleIndex < meshTriangles.length; triangleIndex += 1) {
        const base = triangleIndex * 3;
        includeDensePoint(meshTriangles.indices[base]);
        includeDensePoint(meshTriangles.indices[base + 1]);
        includeDensePoint(meshTriangles.indices[base + 2]);
      }
    } else if (densePoints.length > 0) {
      for (let index = 0; index < densePoints.length; index += 1) {
        includeDensePoint(index);
      }
    } else if (reconstruction.points.length > 0) {
      for (const point of reconstruction.points) {
        includePosition(point.x, point.y, point.z);
      }
    } else {
      for (const camera of displayCameras) {
        includePosition(camera.x, camera.y, camera.z);
      }
    }

    if (positionCount === 0) {
      return { center: [0, 0, 0] as const, extent: 1 };
    }
    const center = [
      (min[0] + max[0]) * 0.5,
      (min[1] + max[1]) * 0.5,
      (min[2] + max[2]) * 0.5,
    ] as const;
    const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1);
    return { center, extent };
  }, [densePoints, displayCameras, meshTriangles, reconstruction.points, renderMode]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);

    const cosYaw = Math.cos(view.yaw);
    const sinYaw = Math.sin(view.yaw);
    const cosPitch = Math.cos(view.pitch);
    const sinPitch = Math.sin(view.pitch);
    const distance = bounds.extent * (2.6 / view.zoom) + 0.5;

    const project = (x: number, y: number, z: number) => {
      const cx = x - bounds.center[0];
      const cy = y - bounds.center[1];
      const cz = z - bounds.center[2];
      const yawX = cx * cosYaw - cz * sinYaw;
      const yawZ = cx * sinYaw + cz * cosYaw;
      const pitchY = cy * cosPitch - yawZ * sinPitch;
      const viewZ = cy * sinPitch + yawZ * cosPitch;
      const depth = viewZ + distance;
      const focal = Math.min(width, height) * 0.9;
      const scale = focal / Math.max(0.08, depth);
      return {
        x: width * 0.5 + yawX * scale,
        y: height * 0.5 - pitchY * scale,
        z: depth,
        scale,
        viewX: yawX,
        viewY: pitchY,
        viewZ,
      };
    };

    type Projected = ReturnType<typeof project>;
    const projectedMeshTriangles: Array<{
      triangleIndex: number;
      projected: [Projected, Projected, Projected];
      depth: number;
      r: number;
      g: number;
      b: number;
      textureReference: number | null;
      textureSource: [Point2, Point2, Point2] | null;
    }> = [];

    for (let triangleIndex = 0; triangleIndex < meshTriangles.length; triangleIndex += 1) {
      const triangleBase = triangleIndex * 3;
      const aIndex = meshTriangles.indices[triangleBase];
      const bIndex = meshTriangles.indices[triangleBase + 1];
      const cIndex = meshTriangles.indices[triangleBase + 2];
      if (
        aIndex >= densePoints.length ||
        bIndex >= densePoints.length ||
        cIndex >= densePoints.length
      ) {
        continue;
      }
      const aBase = aIndex * 4;
      const bBase = bIndex * 4;
      const cBase = cIndex * 4;
      const projectedA = project(
        densePoints.values[aBase],
        densePoints.values[aBase + 1],
        densePoints.values[aBase + 2],
      );
      const projectedB = project(
        densePoints.values[bBase],
        densePoints.values[bBase + 1],
        densePoints.values[bBase + 2],
      );
      const projectedC = project(
        densePoints.values[cBase],
        densePoints.values[cBase + 1],
        densePoints.values[cBase + 2],
      );
      if (projectedA.z <= 0.05 || projectedB.z <= 0.05 || projectedC.z <= 0.05) {
        continue;
      }
      const ab = {
        x: projectedB.viewX - projectedA.viewX,
        y: projectedB.viewY - projectedA.viewY,
        z: projectedB.viewZ - projectedA.viewZ,
      };
      const ac = {
        x: projectedC.viewX - projectedA.viewX,
        y: projectedC.viewY - projectedA.viewY,
        z: projectedC.viewZ - projectedA.viewZ,
      };
      const normal = {
        x: ab.y * ac.z - ab.z * ac.y,
        y: ab.z * ac.x - ab.x * ac.z,
        z: ab.x * ac.y - ab.y * ac.x,
      };
      const normalLength = Math.hypot(normal.x, normal.y, normal.z);
      const facing = normalLength > 1e-9 ? Math.abs(normal.z) / normalLength : 0;
      const light = 0.48 + facing * 0.52;
      const aRgb = aIndex * 3;
      const bRgb = bIndex * 3;
      const cRgb = cIndex * 3;
      const textureReference = triangleTextureReference(
        aIndex,
        bIndex,
        cIndex,
        densePointReferenceFrames,
      );
      const textureSource =
        textureReference !== null && denseGridSites.length === densePoints.length
          ? ([aIndex, bIndex, cIndex].map((index) => ({
              x: denseGridSites.xy[index * 2],
              y: denseGridSites.xy[index * 2 + 1],
            })) as [Point2, Point2, Point2])
          : null;
      projectedMeshTriangles.push({
        triangleIndex,
        projected: [projectedA, projectedB, projectedC],
        depth: (projectedA.z + projectedB.z + projectedC.z) / 3,
        r: clampedChannel(
          ((densePoints.rgb[aRgb] + densePoints.rgb[bRgb] + densePoints.rgb[cRgb]) / 3) * light,
        ),
        g: clampedChannel(
          ((densePoints.rgb[aRgb + 1] +
            densePoints.rgb[bRgb + 1] +
            densePoints.rgb[cRgb + 1]) /
            3) *
            light,
        ),
        b: clampedChannel(
          ((densePoints.rgb[aRgb + 2] +
            densePoints.rgb[bRgb + 2] +
            densePoints.rgb[cRgb + 2]) /
            3) *
            light,
        ),
        textureReference,
        textureSource,
      });
    }
    projectedMeshTriangles.sort((left, right) => right.depth - left.depth);

    for (const item of projectedMeshTriangles) {
      const [a, b, c] = item.projected;
      const traceTriangle = () => {
        context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.lineTo(c.x, c.y);
        context.closePath();
      };
      const textureFrame =
        item.textureReference === null ? null : textureFrames[item.textureReference] ?? null;
      const textureEntry =
        item.textureReference === null ? null : textureImages[item.textureReference] ?? null;
      const textureImage =
        textureFrame && textureEntry?.source === textureFrame.thumbnail
          ? textureEntry.image
          : null;
      const textureTransform =
        textureFrame && textureImage && item.textureSource
          ? affineTriangleTransform(item.textureSource, [
              { x: a.x, y: a.y },
              { x: b.x, y: b.y },
              { x: c.x, y: c.y },
            ])
          : null;

      if (!textureInteractionActive && textureFrame && textureImage && textureTransform) {
        traceTriangle();
        context.save();
        context.clip();
        context.setTransform(...textureTransform);
        context.globalAlpha = 0.8 + meshTriangles.confidence[item.triangleIndex] * 0.2;
        context.drawImage(textureImage, 0, 0, textureFrame.width, textureFrame.height);
        context.restore();
        traceTriangle();
      } else {
        traceTriangle();
        context.fillStyle = `rgba(${item.r}, ${item.g}, ${item.b}, ${0.76 + meshTriangles.confidence[item.triangleIndex] * 0.24})`;
        context.fill();
      }
      context.lineWidth = 0.42 * ratio;
      context.strokeStyle = "rgba(4, 10, 13, 0.22)";
      context.stroke();
    }

    const showEvidence = renderMode === "evidence" || !hasSurfaceModel;
    if (showEvidence) {
      const projectedDensePoints: Array<{ index: number; projected: Projected }> = [];
      for (let index = 0; index < densePoints.length; index += 1) {
        const base = index * 4;
        if (densePoints.values[base + 3] <= 0.05) continue;
        const projected = project(
          densePoints.values[base],
          densePoints.values[base + 1],
          densePoints.values[base + 2],
        );
        if (projected.z > 0.05) projectedDensePoints.push({ index, projected });
      }
      projectedDensePoints.sort((a, b) => b.projected.z - a.projected.z);

      for (const { index, projected } of projectedDensePoints) {
        const valueBase = index * 4;
        const rgbBase = index * 3;
        const confidence = densePoints.values[valueBase + 3];
        const radius = Math.max(0.55 * ratio, Math.min(1.8 * ratio, projected.scale * 0.009));
        context.beginPath();
        context.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
        context.fillStyle = `rgba(${densePoints.rgb[rgbBase]}, ${densePoints.rgb[rgbBase + 1]}, ${densePoints.rgb[rgbBase + 2]}, ${0.14 + confidence * 0.38})`;
        context.fill();
      }

      const projectedPoints = reconstruction.points
        .filter((point) => point.confidence > 0.08)
        .map((point) => ({ point, projected: project(point.x, point.y, point.z) }))
        .filter(({ projected }) => projected.z > 0.05)
        .sort((a, b) => b.projected.z - a.projected.z);

      for (const { point, projected } of projectedPoints) {
        const radius = Math.max(0.7 * ratio, Math.min(2.6 * ratio, projected.scale * 0.012));
        context.beginPath();
        context.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
        context.fillStyle = `rgba(${point.r}, ${point.g}, ${point.b}, ${0.3 + point.confidence * 0.55})`;
        context.fill();
      }
    }

    const showCameras = renderMode === "evidence" || !hasSurfaceModel;
    const projectedCameras = showCameras
      ? displayCameras
          .map((camera) => ({ camera, projected: project(camera.x, camera.y, camera.z) }))
          .filter(({ projected }) => projected.z > 0.05)
      : [];

    if (projectedCameras.length > 0) {
      context.lineWidth = (hasRegisteredGeometry ? 1.5 : 1) * ratio;
      context.strokeStyle = hasRegisteredGeometry
        ? "rgba(111, 220, 255, 0.78)"
        : "rgba(111, 220, 255, 0.38)";
      context.setLineDash(hasRegisteredGeometry ? [] : [4 * ratio, 4 * ratio]);
      context.beginPath();
      projectedCameras.forEach(({ projected }, index) => {
        if (index === 0) context.moveTo(projected.x, projected.y);
        else context.lineTo(projected.x, projected.y);
      });
      context.stroke();
      context.setLineDash([]);
    }

    const hitTargets: CameraHitTarget[] = [];
    for (const { camera, projected } of projectedCameras) {
      const size = Math.max(5 * ratio, Math.min(10 * ratio, projected.scale * 0.032));
      const selected = camera.frame_index === selectedFrameIndex;
      const frameState = cameraState.frames[camera.frame_index];
      const isSeed = frameState?.camera_kind === "seed";
      context.lineWidth = (selected ? 3 : 1.5) * ratio;
      context.strokeStyle = selected
        ? "rgba(111, 220, 255, 1)"
        : isSeed
          ? "rgba(111, 220, 255, 0.95)"
          : hasRegisteredGeometry
            ? "rgba(240, 247, 255, 0.95)"
            : "rgba(174, 201, 213, 0.72)";
      context.fillStyle = selected
        ? "rgba(111, 220, 255, 0.18)"
        : "rgba(111, 220, 255, 0.07)";

      if (hasRegisteredGeometry) {
        if (selected) {
          context.fillRect(
            projected.x - size * 0.7,
            projected.y - size * 0.7,
            size * 1.4,
            size * 1.4,
          );
        }
        context.strokeRect(
          projected.x - size * 0.5,
          projected.y - size * 0.5,
          size,
          size,
        );
        if (isSeed) {
          context.fillStyle = "rgba(111, 220, 255, 0.7)";
          context.fillRect(
            projected.x - 1.5 * ratio,
            projected.y - 1.5 * ratio,
            3 * ratio,
            3 * ratio,
          );
        }
      } else {
        const radius = selected ? size * 0.58 : size * 0.34;
        context.beginPath();
        context.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
        if (selected) context.fill();
        context.stroke();
      }

      hitTargets.push({
        frameIndex: camera.frame_index,
        x: projected.x / ratio,
        y: projected.y / ratio,
        radius: Math.max(12, size / ratio + 7),
      });
    }
    cameraHitTargetsRef.current = hitTargets;
  }, [
    bounds,
    cameraState,
    denseGridSites,
    densePointReferenceFrames,
    densePoints,
    displayCameras,
    hasRegisteredGeometry,
    hasSurfaceModel,
    meshTriangles,
    reconstruction.points,
    renderMode,
    selectedFrameIndex,
    textureFrames,
    textureImages,
    textureInteractionActive,
    view,
  ]);

  useEffect(() => {
    draw();
    const observer = new ResizeObserver(draw);
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [draw]);

  const selectRenderMode = (nextMode: RenderMode) => {
    setRenderMode(nextMode);
    setView((current) => ({ ...current, zoom: 1 }));
  };

  const cameraDiagnostic = hasRegisteredGeometry
    ? `Camera geometry: ${cameraState.calibrated_seed_cameras.length} seed + ${cameraState.registered_cameras.length} additional registered; ${cameraState.dense_eligible_cameras.length} dense-eligible`
    : `Approximate motion only: ${cameraState.approximate_motion_samples.length} sampled poses; no calibrated seed pair was accepted`;

  const denseReferenceAttempts = reconstruction.dense.reference_attempts;
  const acceptedReferenceAttempts = denseReferenceAttempts.filter((attempt) => attempt.accepted);
  const rejectedReferenceAttempts = denseReferenceAttempts.filter((attempt) => !attempt.accepted);
  const rejectedReferenceDetails = rejectedReferenceAttempts
    .map(
      (attempt) =>
        `frame ${attempt.reference_frame}: ${
          attempt.skip_reason ??
          `${attempt.accepted_points} points after ${attempt.sampled_pixels} sampled pixels; reciprocal depth rejected ${attempt.reciprocal_rejected_points} of ${attempt.reciprocal_checked_points}`
        }`,
    )
    .join("; ");
  const referenceCoverageDiagnostic =
    denseReferenceAttempts.length > 0
      ? `Reference coverage: ${acceptedReferenceAttempts.length} of ${denseReferenceAttempts.length} attempted views produced accepted surface patches${
          rejectedReferenceDetails.length > 0 ? `; rejected ${rejectedReferenceDetails}` : ""
        }`
      : "Reference coverage: no multi-reference dense attempt evidence was reported";

  const surfaceCompletionDiagnostic =
    reconstruction.dense.surface_completion_proposals > 0
      ? `Surface completion: ${reconstruction.dense.surface_completed_points} of ${reconstruction.dense.surface_completion_proposals} coherent proposals accepted; rejected ${reconstruction.dense.surface_completion_rejected_texture} at texture, ${reconstruction.dense.surface_completion_rejected_cross_view} at direct cross-view support, ${reconstruction.dense.surface_completion_rejected_reciprocal} at reciprocal depth, ${reconstruction.dense.surface_completion_rejected_fusion} at fusion, and ${reconstruction.dense.surface_completion_rejected_footprint} at final grid footprint`
      : "Surface completion: no empty grid site had enough coherent neighboring depth evidence to make a proposal";
  const denseDiagnostic = reconstruction.dense.skip_reason
    ? `Dense depth skipped: ${reconstruction.dense.skip_reason} · ${referenceCoverageDiagnostic}`
    : reconstruction.dense.accepted_points > 0
      ? `Dense surface evidence: ${reconstruction.dense.accepted_points} accepted scene samples; ${referenceCoverageDiagnostic}; ${surfaceCompletionDiagnostic}; reciprocal depth rejected ${reconstruction.dense.reciprocal_rejected_points} of ${reconstruction.dense.reciprocal_checked_points} candidates and spatial fusion rejected ${reconstruction.dense.fusion_rejected_observations} reverse observations`
      : reconstruction.dense.reciprocal_consistent_points > 0
        ? `Dense depth found ${reconstruction.dense.reciprocal_consistent_points} reciprocal-consistent candidates, but fusion rejected all remaining geometry (${reconstruction.dense.fusion_rejected_observations} inconsistent reverse observations); ${referenceCoverageDiagnostic}; ${surfaceCompletionDiagnostic}`
        : reconstruction.dense.reciprocal_checked_points > 0
          ? `Coarse dense depth ran, but reciprocal depth rejected ${reconstruction.dense.reciprocal_rejected_points} of ${reconstruction.dense.reciprocal_checked_points} candidates after the texture and ambiguity gates; ${referenceCoverageDiagnostic}; ${surfaceCompletionDiagnostic}`
          : `Coarse dense depth ran, but no depth hypothesis passed the texture and ambiguity gates; ${referenceCoverageDiagnostic}; ${surfaceCompletionDiagnostic}`;

  const meshReferencePatches = reconstruction.mesh.reference_patches;
  const acceptedMeshReferencePatches = meshReferencePatches.filter(
    (patch) => patch.accepted_triangles > 0,
  );
  const rejectedMeshReferencePatches = meshReferencePatches.filter(
    (patch) => patch.accepted_triangles === 0,
  );
  const rejectedMeshReferenceDetails = rejectedMeshReferencePatches
    .map((patch) => {
      const reason = patch.skip_reason
        ? patch.skip_reason
        : patch.attempted
          ? `${patch.candidate_triangles} triangle candidates; rejected ${patch.rejected_discontinuities} at discontinuities and ${patch.rejected_degenerate} as degenerate/orientation-flipped`
          : "mesh reconstruction was not attempted";
      return `frame ${patch.reference_frame}: ${reason}`;
    })
    .join("; ");
  const meshCoverageDiagnostic =
    meshReferencePatches.length > 0
      ? `Mesh coverage: ${acceptedMeshReferencePatches.length} of ${meshReferencePatches.length} accepted dense patches produced surface triangles${
          rejectedMeshReferenceDetails.length > 0
            ? `; rejected ${rejectedMeshReferenceDetails}`
            : ""
        }`
      : "Mesh coverage: no per-reference mesh evidence was reported";

  const meshGridRejected = reconstruction.mesh.rejected_grid_vertices;
  const meshAdmissionDiagnostic =
    meshGridRejected > 0 ? `${meshGridRejected} fused points were not admitted to the mesh grid; ` : "";
  const meshDiagnostic = reconstruction.mesh.attempted
    ? reconstruction.mesh.accepted_triangles > 0
      ? `Surface model: ${reconstruction.mesh.accepted_triangles} accepted triangles; ${meshCoverageDiagnostic}; ${meshAdmissionDiagnostic}rejected ${reconstruction.mesh.rejected_discontinuities} discontinuity bridges and ${reconstruction.mesh.rejected_degenerate} degenerate/orientation-flipped candidates`
      : `No surface model passed the mesh gates; ${meshCoverageDiagnostic}; ${meshAdmissionDiagnostic}${reconstruction.mesh.rejected_discontinuities} discontinuity and ${reconstruction.mesh.rejected_degenerate} degenerate/orientation candidates were rejected. The points shown are reconstruction evidence, not the final model.`
    : reconstruction.mesh.skip_reason
      ? `Surface model unavailable: ${reconstruction.mesh.skip_reason}; ${meshCoverageDiagnostic}${
          meshGridRejected > 0
            ? `; ${meshGridRejected} fused points were rejected at mesh-grid admission`
            : ""
        }. The points shown are reconstruction evidence, not the final model.`
      : `Surface model unavailable; ${meshCoverageDiagnostic}. The points shown are reconstruction evidence, not the final model.`;

  const textureDiagnostic = hasSurfaceModel
    ? textureEligibleTriangles > 0
      ? `Texture projection: ${textureEligibleTriangles} of ${meshTriangles.length} accepted triangles retain an unambiguous Rust-owned reference-grid mapping; missing or unloaded reference images fall back to vertex color without changing geometry`
      : "Texture projection: no accepted triangle has an unambiguous single-reference grid mapping, so vertex-color shading is retained without changing geometry"
    : "Texture projection unavailable without accepted surface triangles";

  const primaryDiagnostic =
    hasSurfaceModel && renderMode === "model"
      ? `${meshDiagnostic} · ${textureDiagnostic} · ${denseDiagnostic}`
      : `${cameraDiagnostic} · ${denseDiagnostic} · ${meshDiagnostic}`;

  return (
    <>
      <canvas
        ref={canvasRef}
        className="scene-canvas"
        onPointerDown={(event) => {
          if (textureInteractionTimeoutRef.current !== null) {
            clearTimeout(textureInteractionTimeoutRef.current);
            textureInteractionTimeoutRef.current = null;
          }
          setTextureInteractionActive(true);
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = {
            x: event.clientX,
            y: event.clientY,
            startX: event.clientX,
            startY: event.clientY,
            moved: false,
          };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag) return;
          const totalDistance = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY);
          const moved = drag.moved || totalDistance > 2;
          const dx = event.clientX - (drag.moved ? drag.x : drag.startX);
          const dy = event.clientY - (drag.moved ? drag.y : drag.startY);
          dragRef.current = {
            ...drag,
            x: event.clientX,
            y: event.clientY,
            moved,
          };
          if (!moved) return;
          setView((current) => ({
            ...current,
            yaw: current.yaw + dx * 0.008,
            pitch: Math.max(-1.2, Math.min(1.2, current.pitch + dy * 0.008)),
          }));
        }}
        onPointerUp={(event) => {
          const drag = dragRef.current;
          dragRef.current = null;
          setTextureInteractionActive(false);
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
          if (!drag || drag.moved || !onSelectFrame || renderMode !== "evidence") return;
          const rect = event.currentTarget.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const y = event.clientY - rect.top;
          const selected = cameraHitTargetsRef.current
            .map((target) => ({
              target,
              distance: Math.hypot(target.x - x, target.y - y),
            }))
            .filter(({ target, distance }) => distance <= target.radius)
            .sort((a, b) => a.distance - b.distance)[0];
          if (selected) onSelectFrame(selected.target.frameIndex);
        }}
        onPointerCancel={() => {
          dragRef.current = null;
          setTextureInteractionActive(false);
        }}
        onWheel={(event) => {
          event.preventDefault();
          setTextureInteractionActive(true);
          if (textureInteractionTimeoutRef.current !== null) {
            clearTimeout(textureInteractionTimeoutRef.current);
          }
          textureInteractionTimeoutRef.current = setTimeout(() => {
            textureInteractionTimeoutRef.current = null;
            setTextureInteractionActive(false);
          }, 120);
          setView((current) => ({
            ...current,
            zoom: Math.max(0.45, Math.min(3.5, current.zoom * Math.exp(-event.deltaY * 0.001))),
          }));
        }}
        aria-label="Interactive 3D reconstruction. Drag to orbit and use the mouse wheel to zoom. Switch to evidence mode to inspect camera markers and point evidence."
      />
      {hasSurfaceModel ? (
        <div
          style={{
            position: "absolute",
            left: 14,
            bottom: 52,
            zIndex: 2,
            display: "flex",
            gap: 6,
          }}
          aria-label="Reconstruction display mode"
        >
          <button
            type="button"
            aria-pressed={renderMode === "model"}
            onClick={() => selectRenderMode("model")}
            style={{
              border: "1px solid rgba(111, 220, 255, 0.45)",
              borderRadius: 999,
              padding: "5px 10px",
              background:
                renderMode === "model"
                  ? "rgba(111, 220, 255, 0.18)"
                  : "rgba(4, 10, 13, 0.76)",
              color: "inherit",
              cursor: "pointer",
              fontSize: "0.72rem",
            }}
          >
            Model
          </button>
          <button
            type="button"
            aria-pressed={renderMode === "evidence"}
            onClick={() => selectRenderMode("evidence")}
            style={{
              border: "1px solid rgba(111, 220, 255, 0.3)",
              borderRadius: 999,
              padding: "5px 10px",
              background:
                renderMode === "evidence"
                  ? "rgba(111, 220, 255, 0.18)"
                  : "rgba(4, 10, 13, 0.76)",
              color: "inherit",
              cursor: "pointer",
              fontSize: "0.72rem",
            }}
          >
            Evidence
          </button>
        </div>
      ) : null}
      <div className="viewer-diagnostic" aria-live="polite">
        {primaryDiagnostic}
      </div>
    </>
  );
}
