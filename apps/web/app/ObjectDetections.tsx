import { useEffect, useRef, useState } from "react";
import { loadReconstructionWasm, type SampledFrame } from "../src/reconstruction";

export type PreviewFrame = Pick<SampledFrame, "height" | "thumbnail" | "time" | "width"> & {
  contentHash: string | null;
};

export async function previewFrames(frames: SampledFrame[]): Promise<PreviewFrame[]> {
  return Promise.all(frames.map(async ({ width, height, time, thumbnail, rgba }) => {
    let contentHash = null;
    if (globalThis.crypto?.subtle) {
      const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", rgba));
      contentHash = "sha256:" + Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
    }
    return { width, height, time, thumbnail, contentHash };
  }));
}

// Only fields needed for display. Rust validates the full versioned native artifact.
type Mask = { width: number; height: number; counts: number[] };
type Observation = {
  id: string;
  frame_index: number;
  label: string;
  bbox: [number, number, number, number];
  detection_confidence: number;
  mask: Mask | null;
  mask_confidence: number | null;
};
type Inventory = {
  status: "available" | "unavailable";
  unavailable_reason?: string;
  frames: { frame_index: number; width: number; height: number; content_hash: string }[];
  observations: Observation[];
  tracks: { id: string; label: string; observations: string[] }[];
  uncertainties: { frame_index: number; observation: string | null; reason: string }[];
};

export type ImportedInventory = { inventory: Inventory; json: string };

export async function readInventory(json: string, frames: PreviewFrame[]): Promise<ImportedInventory> {
  const value: unknown = JSON.parse(json);
  const wasm = await loadReconstructionWasm();
  wasm.validate_semantic_inventory(value);
  const inventory = value as Inventory;
  if (!inventory.frames.length) {
    throw new Error("The inventory contains no source frames to match against this video.");
  }
  for (const recorded of inventory.frames) {
    const source = frames[recorded.frame_index];
    if (!source || source.width !== recorded.width || source.height !== recorded.height ||
        source.contentHash !== recorded.content_hash) {
      throw new Error(`Sampled frame ${recorded.frame_index + 1} differs from the inventory. Open the same video with the same sampling settings.`);
    }
  }
  return { inventory, json };
}

function SelectedMask({ mask }: { mask: Mask }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = ref.current?.getContext("2d");
    if (!context) { return; }
    const pixels = context.createImageData(mask.width, mask.height);
    let offset = 0;
    const tint = [255, 194, 72, 110];
    for (const [index, count] of mask.counts.entries()) {
      if (index % 2 === 1) {
        for (let pixel = offset; pixel < offset + count; pixel++) {
          pixels.data.set(tint, pixel * 4);
        }
      }
      offset += count;
    }
    context.putImageData(pixels, 0, 0);
  }, [mask]);
  return <canvas ref={ref} className="object-mask" width={mask.width} height={mask.height} aria-hidden="true" />;
}

const percentage = (confidence: number) => `${Math.round(confidence * 100)}%`;

export function ObjectDetections({
  frames, imported, onImport, frameIndex, onSelectFrame, busy, detecting, detectionError, onSave,
}: {
  frames: PreviewFrame[];
  imported: ImportedInventory | null;
  onImport: (value: ImportedInventory) => void;
  frameIndex: number | null;
  onSelectFrame: (index: number) => void;
  busy: boolean;
  detecting: boolean;
  detectionError: string;
  onSave: () => void;
}) {
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [labelFilter, setLabelFilter] = useState("");
  const [showOtherBoxes, setShowOtherBoxes] = useState(false);
  const inventory = imported?.inventory;
  const index = frameIndex ?? inventory?.observations[0]?.frame_index ?? 0;
  const frame = frames[index];
  const labels = Array.from(new Set(inventory?.observations.map((o) => o.label))).sort();
  const frameObjects = inventory?.observations.filter((o) => o.frame_index === index) ?? [];
  const observations = frameObjects.filter((o) => !labelFilter || o.label === labelFilter)
    .sort((a, b) => b.detection_confidence - a.detection_confidence);
  const preferredTrack = selectedId && inventory?.tracks.find((t) => t.observations.includes(selectedId));
  const selected = preferredTrack
    ? observations.find((o) => preferredTrack.observations.includes(o.id))
    : observations.find((o) => o.id === selectedId) ?? observations[0];
  const track = selected && inventory?.tracks.find((t) => t.observations.includes(selected.id));
  const canMatch = Boolean(frames[0]?.contentHash);
  const recorded = inventory?.frames.some((f) => f.frame_index === index);

  async function importFile(file: File) {
    setLoading(true);
    setError("");
    try {
      const result = await readInventory(await file.text(), frames);
      onImport(result);
      setSelectedId(null);
      setLabelFilter("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="section-block" aria-label="Detected objects">
      <div className="section-heading">
        <h2>Detected objects</h2>
        <div className="input-actions">
          {inventory && <button className="demo-button" type="button" onClick={onSave}>Save object detections</button>}
          <label className="upload-button">
            Import object detections
            <input type="file" accept=".json,application/json" disabled={busy || detecting || loading || !canMatch}
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = "";
                if (file) { void importFile(file); }
              }} />
          </label>
        </div>
      </div>
      {!canMatch && <p>Open the app on localhost or HTTPS to match detections to the sampled pixels.</p>}
      {error && <p role="alert" className="notice notice-error">{error}</p>}
      {detecting && <p role="status">Detecting objects on this device…</p>}
      {detectionError && <p role="alert" className="notice notice-error">{detectionError}</p>}
      {!inventory && !detecting && !detectionError && <p>Run the local Studio to detect objects automatically, or import a .objects.json file for this video.</p>}
      {inventory && <p data-testid="object-inventory-loaded" data-status={inventory.status} role="status">
        {inventory.status === "unavailable" ? `Object detection unavailable. ${inventory.unavailable_reason}`
          : `${inventory.observations.length} detections across ${inventory.frames.length} analyzed frames · ${inventory.tracks.length} candidate tracks`}
      </p>}
      {inventory?.status === "available" && frame && (
        <>
          <div className="object-controls">
            <label>Source frame
              <select value={index} onChange={(event) => onSelectFrame(Number(event.target.value))}>
                {frames.map((f, i) => <option key={i} value={i}>Frame {i + 1} · {f.time.toFixed(2)}s</option>)}
              </select>
            </label>
            <label>Object class
              <select value={labelFilter} onChange={(event) => {
                setLabelFilter(event.target.value);
                setSelectedId(null);
              }}>
                <option value="">All objects</option>
                {labels.map((label) => <option key={label} value={label}>{label}</option>)}
              </select>
            </label>
            <label><input type="checkbox" checked={showOtherBoxes}
              onChange={(event) => setShowOtherBoxes(event.target.checked)} /> Show other boxes</label>
          </div>
          <div className="object-inspector">
            <div className="object-image">
              <img src={frame.thumbnail} width={frame.width} height={frame.height} alt={`Source video frame ${index + 1} with object detections`} />
              {selected?.mask && <SelectedMask mask={selected.mask} />}
              <svg viewBox={`0 0 ${frame.width} ${frame.height}`} aria-hidden="true">
                {observations.filter((o) => showOtherBoxes || o.id === selected?.id).map((item) => {
                  const [x, y, right, bottom] = item.bbox;
                  return <rect key={item.id} x={x} y={y} width={right - x} height={bottom - y}
                    className={selected?.id === item.id ? "selected" : ""}
                    onClick={() => setSelectedId(item.id)}>
                    <title>{item.label} {percentage(item.detection_confidence)}</title>
                  </rect>;
                })}
                {selected && <text x={Math.max(2, Math.min(selected.bbox[0], frame.width - 70))}
                  y={Math.max(12, selected.bbox[1] - 2)}>{selected.label} {percentage(selected.detection_confidence)}</text>}
              </svg>
            </div>
            <div className="object-list">
              {!recorded && <p>The detector did not analyze this frame.</p>}
              {recorded && frameObjects.length === 0 && <p>No objects detected in this frame.</p>}
              {frameObjects.length > 0 && observations.length === 0 && <p>No {labelFilter} detections in this frame.</p>}
              <div className="object-options">
                {observations.map((item, ordinal) => <button type="button" key={item.id}
                  aria-pressed={selected?.id === item.id} onClick={() => setSelectedId(item.id)}>
                  {ordinal + 1}. {item.label} · {percentage(item.detection_confidence)}
                </button>)}
              </div>
              {recorded && preferredTrack && !selected && <p>The selected {preferredTrack.label} track has no detection in this frame.</p>}
              {selected && <p>{selected.mask ? "Pixel mask" : "Bounding box only"}
                {selected.mask_confidence != null && ` · Mask score ${percentage(selected.mask_confidence)}`}
                {track ? ` · Tracked in ${track.observations.length} sampled frame(s)` : " · Untracked detection"}
              </p>}
              {selected && inventory.uncertainties.filter((item) => item.observation === selected.id)
                .map((item) => <p key={item.reason}>{item.reason}</p>)}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
