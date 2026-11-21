import { reconstructFrames, type ReconstructionResult, type SampledFrame } from "./reconstruction";

export const localStudio = () => document.querySelector('meta[name="video-to-3d-local-studio"]') !== null &&
  location.hostname === "127.0.0.1";

function sampledFramesForm(frames: SampledFrame[]) {
  const form = new FormData();
  form.append("frames", JSON.stringify(frames.map(({ width, height, time }) => ({ width, height, time }))));
  for (const frame of frames) {
    form.append("pixels", new Blob([frame.rgba]));
  }
  return form;
}

async function post(path: string, body: BodyInit) {
  const response = await fetch(path, { method: "POST", body, redirect: "error" });
  if (!response.ok) {
    throw new Error(await response.text() || `Local processing failed (HTTP ${response.status})`);
  }
  return response;
}

export async function reconstructStudioFrames(
  frames: SampledFrame[],
  onStatus: (message: string) => void,
  detectedInventory: () => Promise<string | null>,
  onRegistered: (result: ReconstructionResult) => void,
) {
  if (!localStudio()) { return { result: await reconstructFrames(frames), nativeSurface: null }; }
  onStatus("Registering cameras with local COLMAP…");
  let result: ReconstructionResult;
  let registration: string | null;
  let incrementalViews: number | undefined;
  try {
    const response = await post("/local/register-cameras", sampledFramesForm(frames));
    registration = response.headers.get("X-Video-To-3D-Registration");
    const evidence: unknown = await response.json();
    result = await reconstructFrames(frames, evidence);
    if (!result.sparse_camera_provider) { throw new Error("Missing validated sparse camera provenance"); }
    if (evidence && typeof evidence === "object" && "registration_diagnostics" in evidence) {
      const diagnostics = evidence.registration_diagnostics;
      if (diagnostics && typeof diagnostics === "object" &&
          "mapper" in diagnostics && diagnostics.mapper === "global" &&
          "incremental_registered_frames" in diagnostics) {
        const count = diagnostics.incremental_registered_frames;
        if (typeof count === "number" && Number.isSafeInteger(count) &&
            count > 0 && count < result.accepted_camera_evidence.length) {
          incrementalViews = count;
        }
      }
    }
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    onStatus(`Using browser camera reconstruction. Local COLMAP registration failed. ${detail}`);
    return { result: await reconstructFrames(frames), nativeSurface: null };
  }
  const acceptedViews = result.accepted_camera_evidence.length;
  const status = incrementalViews === undefined
    ? `Local COLMAP registered ${acceptedViews} of ${frames.length} cameras.`
    : `Global COLMAP registration increased coverage from ${incrementalViews} to ${acceptedViews} of ${frames.length} cameras.`;
  if (registration) {
    onRegistered(result);
    try {
      onStatus(status + " Reconstructing dense geometry on the GPU…");
      const inventory = await detectedInventory();
      const response = await post("/local/dense/" + encodeURIComponent(registration), inventory ?? "");
      const denseTriangles = Number(response.headers.get("X-Video-To-3D-Dense-Triangles"));
      if (!Number.isSafeInteger(denseTriangles) || denseTriangles <= 0) {
        throw new Error("COLMAP did not report a validated dense surface.");
      }
      if (denseTriangles < result.mesh_triangles.length) {
        throw new Error(`COLMAP recovered ${denseTriangles} triangles. Rust recovered ${result.mesh_triangles.length} triangles from the same registered cameras.`);
      }
      const nativeSurface = await response.blob();
      const report = response.headers.get("X-Video-To-3D-Dense-Report");
      onStatus(status + " COLMAP dense geometry passed Rust validation." + (report ? " " + report : ""));
      return { result, nativeSurface };
    } catch (cause) {
      onStatus(status + " Using Rust surfaces. " + (cause instanceof Error ? cause.message : String(cause)));
      return { result, nativeSurface: null };
    }
  }
  onStatus(status);
  return { result, nativeSurface: null };
}

/** Only the loopback Studio host advertises the on-device inference service. */
export async function detectInLocalStudio(frames: SampledFrame[]): Promise<string | null> {
  if (!localStudio()) { return null; }
  return (await post("/local/objects", sampledFramesForm(frames))).text();
}

export type ModelCandidate = {
  method: "multiview" | "triposr";
  object: string;
  label: string;
  frame: number | null;
  crop: [number, number, number, number] | null;
  triangles: number;
  meshUrl: string | null;
  placement: "accepted" | "approximate" | "contradicted" | "unplaced";
  reason: string;
};

export type ObjectModels = {
  token: string;
  candidates: ModelCandidate[];
  unavailableReason?: string;
  unpackaged: number;
};

export async function reconstructLocalObjects(frames: SampledFrame[], inventory: string, result: ReconstructionResult): Promise<ObjectModels | null> {
  if (!localStudio()) { return null; }
  const form = sampledFramesForm(frames);
  const points: number[][] = [];
  if (result.calibrated_pair) {
    for (const p of result.points) { points.push([p.x, p.y, p.z]); }
    const xyz = result.dense_points.values;
    for (const region of result.geometry_confidence.regions) {
      if (region.origin !== "geometric_multi_view") { continue; }
      for (let i = region.points.start; i < region.points.start + region.points.count; i++) {
        points.push([xyz[i * 4], xyz[i * 4 + 1], xyz[i * 4 + 2]]);
      }
    }
  }
  const cameras = result.accepted_camera_evidence.map(({ frame_index, rotation, translation }) => {
    const frame = frames[frame_index];
    return { frame_index, rotation, translation, focal_pixels: result.calibrated_pair?.focal_pixels,
      principal_point: [frame.width / 2, frame.height / 2] };
  });
  form.append("inventory", new Blob([inventory], { type: "application/json" }));
  form.append("geometry", new Blob([JSON.stringify({ cameras, points })], { type: "application/json" }));
  return (await post("/local/models", form)).json();
}

export async function completeLocalScene(observed: Blob, models: ObjectModels) {
  return (await post("/local/complete/" + encodeURIComponent(models.token), observed)).blob();
}
