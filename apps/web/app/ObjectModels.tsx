import { createElement, useEffect, useId, useState } from "react";
import type { PreviewFrame } from "./ObjectDetections";
import type { ReconstructionResult } from "../src/reconstruction";
import { objectCameraView, type CameraView } from "../src/sourceView";
import type { ObjectModels as ModelRun } from "../src/localObjects";

export function MeshViewer({ url, alt, camera, generatorFrame = false }: { url: string; alt: string; camera?: CameraView; generatorFrame?: boolean }) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    void import("@google/model-viewer").then(() => {
      if (active) { setReady(true); }
    }).catch((cause: unknown) => {
      if (active) { setError(cause instanceof Error ? cause.message : String(cause)); }
    });
    return () => { active = false; };
  }, []);
  if (error) { return <p role="alert">Model preview unavailable. {error}</p>; }
  if (!ready) { return <p role="status">Loading mesh viewer…</p>; }
  // TripoSR stores +Z up and +X toward the source camera; glTF viewers use +Y up.
  return createElement("model-viewer", { src: url, alt, orientation: generatorFrame ? "0deg -90deg -90deg" : "0deg 0deg 0deg",
    "camera-orbit": camera?.orbit, "camera-target": camera?.target, "field-of-view": camera?.fov,
    "min-camera-orbit": "auto auto 0.001m", "camera-controls": true, "touch-action": "pan-y", style: { display: "block", width: "100%" },
    onError: () => setError("The browser could not display this mesh.") });
}

export function ObjectModels({ result, running, error, frames, scene, onFrame }: {
  result: ModelRun | null; frames: PreviewFrame[]; scene: ReconstructionResult | null; running: boolean; error: string; onFrame: (index: number) => void;
}) {
  const [selected, select] = useState("");
  const [label, setLabel] = useState("");
  const clip = useId();
  if (!result && !running && !error) { return null; }
  const key = (c: ModelRun["candidates"][number]) => c.method + "/" + c.object;
  const labels = [...new Set(result?.candidates.map(c => c.label))].sort();
  const options = result?.candidates.filter(c => !label || c.label === label) ?? [];
  const candidate = options.find(c => key(c) === selected) ??
    options.find(c => c.placement === "accepted" && c.meshUrl) ?? options.find(c => c.meshUrl) ?? options[0];
  const generated = result?.candidates.filter(c => c.meshUrl).length ?? 0;
  const accepted = new Set(result?.candidates.filter(c => c.placement === "accepted" && c.meshUrl).map(c => c.object)).size;
  const frame = candidate?.frame != null ? frames[candidate.frame] : undefined;
  const crop = candidate?.crop;
  const camera = scene?.accepted_camera_evidence.find(c => c.frame_index === candidate?.frame);
  const sourceView = candidate?.method === "multiview" && frame && crop && camera && scene?.calibrated_pair
    ? objectCameraView(camera, crop, frame, scene.calibrated_pair.focal_pixels) : undefined;
  return <section className="section-block" aria-label="Reconstructed objects">
    <h2>Reconstructed objects</h2>
    {running && <p role="status">Reconstructing detected objects from their masked video references…</p>}
    {error && <p role="alert" className="notice notice-error">{error}</p>}
    {result && <p role="status" data-testid="object-models-ready" data-generated={generated} data-accepted={accepted}>
      {generated} candidate meshes · {accepted} accepted scene placements
      {result.unpackaged > 0 && ` · ${result.unpackaged} identities without suitable references`}
    </p>}
    {generated > 0 && result && <a className="demo-button" href={`/local/models/${result.token}/library.glb`}
      download="object-library.glb">Download object library (unplaced grid)</a>}
    {result?.unavailableReason && <p>{result.unavailableReason}</p>}
    {labels.length > 1 && <label className="object-frame-choice">Object class
      <select value={label} onChange={event => setLabel(event.target.value)}>
        <option value="">All classes</option>
        {labels.map(value => <option key={value} value={value}>{value}</option>)}
      </select>
    </label>}
    {candidate && result && <>
      <label className="object-frame-choice">Object model
        <select value={key(candidate)} onChange={event => select(event.target.value)}>
          {options.map((c, index) => <option key={key(c)} value={key(c)}>{index + 1}. {c.label} · {c.method === "multiview" ? "Multi-view silhouettes" : "TripoSR"} · {c.meshUrl ? c.placement : "unavailable"}</option>)}
        </select>
      </label>
      <p>{candidate.method === "triposr" ? "Generated from a masked source view." : "Shape estimated from calibrated silhouettes."} {candidate.triangles.toLocaleString()} triangles. {candidate.reason}</p>
      <div className="object-model-comparison">
        {frame && crop && <figure>
          <svg viewBox={`${crop[0]} ${crop[1]} ${crop[2]-crop[0]} ${crop[3]-crop[1]}`}
            role="img" aria-label={`Source crop of ${candidate.label}`}>
            <defs><clipPath id={clip}>
              <rect x={crop[0]} y={crop[1]} width={crop[2] - crop[0]} height={crop[3] - crop[1]} />
            </clipPath></defs>
            <image href={frame.thumbnail} width={frame.width} height={frame.height} clipPath={`url(#${clip})`} />
          </svg>
          <figcaption>Source crop · {frame.time.toFixed(2)} s</figcaption>
        </figure>}
        {candidate.meshUrl && <MeshViewer key={candidate.meshUrl} url={candidate.meshUrl} camera={sourceView} generatorFrame={candidate.method === "triposr"} alt={`3D candidate for ${candidate.label}`}/>}
      </div>
      <div className="input-actions">
        {candidate.meshUrl && <a className="demo-button" href={candidate.meshUrl} download={`${candidate.label}-${candidate.object}-${candidate.method}.glb`}>Download object GLB</a>}
        {candidate.frame !== null && <button type="button" className="demo-button" onClick={() => onFrame(candidate.frame!)}>Show source frame {candidate.frame + 1}</button>}
      </div>
    </>}
  </section>;
}
