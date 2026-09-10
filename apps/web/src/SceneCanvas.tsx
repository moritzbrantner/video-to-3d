"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReconstructionResult } from "./reconstruction";

type SceneCanvasProps = {
  reconstruction: ReconstructionResult;
};

type ViewState = {
  yaw: number;
  pitch: number;
  zoom: number;
};

export function SceneCanvas({ reconstruction }: SceneCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [view, setView] = useState<ViewState>({ yaw: -0.45, pitch: 0.18, zoom: 1 });

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
      const radius = Math.max(
        0.7 * ratio,
        Math.min(2.6 * ratio, projected.scale * 0.012),
      );
      context.beginPath();
      context.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
      context.fillStyle = `rgba(${point.r}, ${point.g}, ${point.b}, ${0.35 + point.confidence * 0.65})`;
      context.fill();
    }

    context.lineWidth = 1.5 * ratio;
    context.strokeStyle = "rgba(111, 220, 255, 0.9)";
    context.beginPath();
    reconstruction.cameras.forEach((camera, index) => {
      const projected = project(camera.x, camera.y, camera.z);
      if (index === 0) context.moveTo(projected.x, projected.y);
      else context.lineTo(projected.x, projected.y);
    });
    context.stroke();

    for (const camera of reconstruction.cameras) {
      const projected = project(camera.x, camera.y, camera.z);
      const size = Math.max(3 * ratio, Math.min(7 * ratio, projected.scale * 0.025));
      context.strokeStyle = "rgba(240, 247, 255, 0.95)";
      context.strokeRect(
        projected.x - size * 0.5,
        projected.y - size * 0.5,
        size,
        size,
      );
    }
  }, [bounds, reconstruction, view]);

  useEffect(() => {
    draw();
    const observer = new ResizeObserver(draw);
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [draw]);

  const denseDiagnostic = reconstruction.dense.skip_reason
    ? `Dense depth skipped: ${reconstruction.dense.skip_reason}`
    : reconstruction.dense.accepted_points > 0
      ? `Coarse dense depth: ${reconstruction.dense.accepted_points} accepted samples from ${reconstruction.dense.source_views} source view${reconstruction.dense.source_views === 1 ? "" : "s"}`
      : "Coarse dense depth ran, but no depth hypothesis passed the texture and ambiguity gates";

  return (
    <>
      <canvas
        ref={canvasRef}
        className="scene-canvas"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { x: event.clientX, y: event.clientY };
        }}
        onPointerMove={(event) => {
          if (!dragRef.current) return;
          const dx = event.clientX - dragRef.current.x;
          const dy = event.clientY - dragRef.current.y;
          dragRef.current = { x: event.clientX, y: event.clientY };
          setView((current) => ({
            ...current,
            yaw: current.yaw + dx * 0.008,
            pitch: Math.max(-1.2, Math.min(1.2, current.pitch + dy * 0.008)),
          }));
        }}
        onPointerUp={() => {
          dragRef.current = null;
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
        onWheel={(event) => {
          event.preventDefault();
          setView((current) => ({
            ...current,
            zoom: Math.max(
              0.45,
              Math.min(3.5, current.zoom * Math.exp(-event.deltaY * 0.001)),
            ),
          }));
        }}
        aria-label="Interactive 3D reconstruction with sparse landmarks, accepted coarse dense depth samples, and registered cameras. Drag to orbit and use the mouse wheel to zoom."
      />
      <div
        aria-live="polite"
        style={{
          position: "absolute",
          top: 14,
          left: 16,
          right: 16,
          color: "var(--muted)",
          fontSize: "0.76rem",
          lineHeight: 1.4,
          pointerEvents: "none",
        }}
      >
        {denseDiagnostic}
      </div>
    </>
  );
}
