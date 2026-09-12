"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReconstructionResult } from "./reconstruction";

type SceneCanvasProps = {
  reconstruction: ReconstructionResult;
  selectedFrameIndex?: number | null;
  onSelectFrame?: (frameIndex: number) => void;
};

type ViewState = {
  yaw: number;
  pitch: number;
  zoom: number;
};

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

export function SceneCanvas({
  reconstruction,
  selectedFrameIndex = null,
  onSelectFrame,
}: SceneCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const cameraHitTargetsRef = useRef<CameraHitTarget[]>([]);
  const [view, setView] = useState<ViewState>({ yaw: -0.45, pitch: 0.18, zoom: 1 });
  const hasRegisteredGeometry = reconstruction.calibrated_pair !== null;

  const bounds = useMemo(() => {
    const positions = [
      ...reconstruction.points.map((point) => [point.x, point.y, point.z] as const),
      ...reconstruction.dense_points.map((point) => [point.x, point.y, point.z] as const),
      ...reconstruction.cameras.map((camera) => [camera.x, camera.y, camera.z] as const),
    ];
    if (positions.length === 0) {
      return { center: [0, 0, 0] as const, extent: 1 };
    }
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const position of positions) {
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], position[axis]);
        max[axis] = Math.max(max[axis], position[axis]);
      }
    }
    const center = [
      (min[0] + max[0]) * 0.5,
      (min[1] + max[1]) * 0.5,
      (min[2] + max[2]) * 0.5,
    ] as const;
    const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1);
    return { center, extent };
  }, [reconstruction]);

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
      const pitchZ = cy * sinPitch + yawZ * cosPitch + distance;
      const focal = Math.min(width, height) * 0.9;
      const scale = focal / Math.max(0.08, pitchZ);
      return {
        x: width * 0.5 + yawX * scale,
        y: height * 0.5 - pitchY * scale,
        z: pitchZ,
        scale,
      };
    };

    const projectedMeshTriangles = reconstruction.mesh_triangles
      .flatMap((triangle) => {
        const a = reconstruction.dense_points[triangle.a];
        const b = reconstruction.dense_points[triangle.b];
        const c = reconstruction.dense_points[triangle.c];
        if (!a || !b || !c) return [];
        const projectedA = project(a.x, a.y, a.z);
        const projectedB = project(b.x, b.y, b.z);
        const projectedC = project(c.x, c.y, c.z);
        if (projectedA.z <= 0.05 || projectedB.z <= 0.05 || projectedC.z <= 0.05) {
          return [];
        }
        return [
          {
            triangle,
            projected: [projectedA, projectedB, projectedC] as const,
            depth: (projectedA.z + projectedB.z + projectedC.z) / 3,
            r: Math.round((a.r + b.r + c.r) / 3),
            g: Math.round((a.g + b.g + c.g) / 3),
            b: Math.round((a.b + b.b + c.b) / 3),
          },
        ];
      })
      .sort((left, right) => right.depth - left.depth);

    for (const item of projectedMeshTriangles) {
      const [a, b, c] = item.projected;
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.lineTo(c.x, c.y);
      context.closePath();
      context.fillStyle = `rgba(${item.r}, ${item.g}, ${item.b}, ${0.05 + item.triangle.confidence * 0.18})`;
      context.fill();
      context.lineWidth = 0.45 * ratio;
      context.strokeStyle = `rgba(${item.r}, ${item.g}, ${item.b}, ${0.06 + item.triangle.confidence * 0.1})`;
      context.stroke();
    }

    const projectedDensePoints = reconstruction.dense_points
      .filter((point) => point.confidence > 0.05)
      .map((point) => ({ point, projected: project(point.x, point.y, point.z) }))
      .filter(({ projected }) => projected.z > 0.05)
      .sort((a, b) => b.projected.z - a.projected.z);

    for (const { point, projected } of projectedDensePoints) {
      const radius = Math.max(0.55 * ratio, Math.min(1.8 * ratio, projected.scale * 0.009));
      context.beginPath();
      context.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
      context.fillStyle = `rgba(${point.r}, ${point.g}, ${point.b}, ${0.18 + point.confidence * 0.48})`;
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
      context.fillStyle = `rgba(${point.r}, ${point.g}, ${point.b}, ${0.35 + point.confidence * 0.65})`;
      context.fill();
    }

    const projectedCameras = reconstruction.cameras
      .map((camera) => ({ camera, projected: project(camera.x, camera.y, camera.z) }))
      .filter(({ projected }) => projected.z > 0.05);

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
      context.lineWidth = (selected ? 3 : 1.5) * ratio;
      context.strokeStyle = selected
        ? "rgba(111, 220, 255, 1)"
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
  }, [bounds, hasRegisteredGeometry, reconstruction, selectedFrameIndex, view]);

  useEffect(() => {
    draw();
    const observer = new ResizeObserver(draw);
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [draw]);

  const cameraDiagnostic = hasRegisteredGeometry
    ? `Registered geometry: ${reconstruction.cameras.length} accepted camera poses`
    : `Approximate motion track only: ${reconstruction.cameras.length} sampled poses; no calibrated seed pair was accepted`;

  const denseDiagnostic = reconstruction.dense.skip_reason
    ? `Dense depth skipped: ${reconstruction.dense.skip_reason}`
    : reconstruction.dense.accepted_points > 0
      ? `Fused dense depth: ${reconstruction.dense.accepted_points} scene points from ${reconstruction.dense.fusion_input_observations} reciprocal-consistent multi-view observations; reciprocal depth rejected ${reconstruction.dense.reciprocal_rejected_points} of ${reconstruction.dense.reciprocal_checked_points} primary candidates and spatial fusion rejected ${reconstruction.dense.fusion_rejected_observations} reverse observations`
      : reconstruction.dense.reciprocal_consistent_points > 0
        ? `Dense depth found ${reconstruction.dense.reciprocal_consistent_points} reciprocal-consistent primary candidates, but fusion rejected all remaining geometry (${reconstruction.dense.fusion_rejected_observations} inconsistent reverse observations)`
        : reconstruction.dense.reciprocal_checked_points > 0
          ? `Coarse dense depth ran, but reciprocal depth rejected ${reconstruction.dense.reciprocal_rejected_points} of ${reconstruction.dense.reciprocal_checked_points} primary candidates after the texture and ambiguity gates`
          : "Coarse dense depth ran, but no depth hypothesis passed the texture and ambiguity gates";

  const meshGridRejected = Math.max(
    0,
    reconstruction.dense_points.length - reconstruction.mesh.grid_vertices,
  );
  const meshAdmissionDiagnostic =
    meshGridRejected > 0 ? `${meshGridRejected} fused points were not admitted to the mesh grid; ` : "";
  const meshDiagnostic = reconstruction.mesh.attempted
    ? reconstruction.mesh.accepted_triangles > 0
      ? `Mesh: ${reconstruction.mesh.accepted_triangles} triangles; ${meshAdmissionDiagnostic}rejected ${reconstruction.mesh.rejected_discontinuities} discontinuity bridges and ${reconstruction.mesh.rejected_degenerate} degenerate/orientation-flipped candidates`
      : `Mesh ran, but no neighboring fused samples formed a continuous triangle; ${meshAdmissionDiagnostic}${reconstruction.mesh.rejected_discontinuities} discontinuity and ${reconstruction.mesh.rejected_degenerate} degenerate/orientation candidates were rejected`
    : reconstruction.mesh.skip_reason
      ? `Mesh skipped: ${reconstruction.mesh.skip_reason}`
      : null;

  return (
    <>
      <canvas
        ref={canvasRef}
        className="scene-canvas"
        onPointerDown={(event) => {
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
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
          if (!drag || drag.moved || !onSelectFrame) return;
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
        }}
        onWheel={(event) => {
          event.preventDefault();
          setView((current) => ({
            ...current,
            zoom: Math.max(0.45, Math.min(3.5, current.zoom * Math.exp(-event.deltaY * 0.001))),
          }));
        }}
        aria-label="Interactive 3D reconstruction. Drag to orbit, use the mouse wheel to zoom, or click a camera marker to select its source frame."
      />
      <div className="viewer-diagnostic" aria-live="polite">
        {cameraDiagnostic} · {denseDiagnostic}{meshDiagnostic ? ` · ${meshDiagnostic}` : ""}
      </div>
    </>
  );
}
