import type { EvidenceCamera, ReconstructionResult } from "./reconstruction";

export type CameraView = { orbit: string; target?: string; fov?: string };

/** Orbit from source camera toward a pixel, accounting for the CV→glTF axis flip. */
function direction(camera: EvidenceCamera, u: number, v: number) {
  const r = camera.rotation;
  return [0, 1, 2].map(k => (k === 0 ? 1 : -1) * (r[k] * u + r[k + 3] * v + r[k + 6]));
}

function orbit(forward: number[], radius: string) {
  return `${Math.atan2(-forward[0], -forward[2])}rad ${Math.acos(Math.max(-1, Math.min(1, -forward[1] / Math.hypot(...forward))))}rad ${radius}`;
}

export function objectCameraView(camera: EvidenceCamera, crop: number[], frame: {width: number; height: number}, focal: number): CameraView {
  const u = ((crop[0] + crop[2]) / 2 - frame.width / 2) / focal;
  const v = ((crop[1] + crop[3]) / 2 - frame.height / 2) / focal;
  return { orbit: orbit(direction(camera, u, v), "auto") };
}

type ViewSource = Pick<ReconstructionResult, "accepted_camera_evidence"> & {
  calibrated_pair: { focal_pixels: number } | null;
  points: { x: number; y: number; z: number }[];
};

export function sceneCameraView(result: ViewSource, frames: {height: number}[], selected: number | null): CameraView | undefined {
  const camera = result.accepted_camera_evidence.find(c => c.frame_index === selected) ?? result.accepted_camera_evidence[0];
  if (!camera || !result.calibrated_pair || !frames[camera.frame_index]) { return undefined; }
  const r = camera.rotation, t = camera.translation;
  const depths = result.points.map(p => r[6] * p.x + r[7] * p.y + r[8] * p.z + t[2])
    .filter(z => z > 0 && Number.isFinite(z)).sort((a,b) => a-b);
  if (!depths.length) { return undefined; }
  // An orbit around a point on the reference camera's optical axis starts at the
  // filmer's viewpoint, even when distant reconstruction outliers expand the bounds.
  const radius = depths[Math.floor(depths.length / 2)];
  const forward = direction(camera, 0, 0);
  const center = [0,1,2].map(k => (k === 0 ? -1 : 1) * (r[k] * t[0] + r[k+3] * t[1] + r[k+6] * t[2]));
  return { orbit: orbit(forward, `${radius}m`), target: center.map((c,k) => `${c + radius * forward[k]}m`).join(" "),
    fov: `${2 * Math.atan(frames[camera.frame_index].height / (2 * result.calibrated_pair.focal_pixels))}rad` };
}
