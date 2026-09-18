"use client";

import { type ChangeEvent, useState } from "react";
import {
  reconstructFrames,
  type FrameCameraState,
  type ReconstructionResult,
  type SampledFrame,
} from "../src/reconstruction";
import { SceneCanvas } from "../src/SceneCanvas";
import { sampleVideo } from "../src/video";

type RunPhase = "queued" | "sampling" | "reconstructing" | "done" | "error";
type StatusPhase = "idle" | RunPhase;
type PreviewFrame = Pick<SampledFrame, "height" | "thumbnail" | "time" | "width">;

type VideoRun = {
  id: string;
  fileName: string;
  phase: RunPhase;
  frames: PreviewFrame[];
  reconstruction: ReconstructionResult | null;
  error: string;
  samplingFps: number;
  frameCap: number;
};

const MIN_SAMPLING_FPS = 0.25;
const MAX_SAMPLING_FPS = 8;
const MIN_FRAME_CAP = 4;
const MAX_FRAME_CAP = 60;

const TREVI_DEMO_URL =
  "https://upload.wikimedia.org/wikipedia/commons/a/a8/Fontaine_de_Trevi.webm";
const TREVI_DEMO_FILE_PAGE =
  "https://commons.wikimedia.org/wiki/File:Fontaine_de_Trevi.webm";
const TREVI_DEMO_LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/";
const TREVI_DEMO_FILE_NAME = "trevi-fountain-demo.webm";

function previewFrames(frames: SampledFrame[]): PreviewFrame[] {
  return frames.map((frame) => ({
    width: frame.width,
    height: frame.height,
    thumbnail: frame.thumbnail,
    time: frame.time,
  }));
}

function normalizedSamplingFps(value: number): number {
  if (!Number.isFinite(value)) return 1.25;
  return Math.min(MAX_SAMPLING_FPS, Math.max(MIN_SAMPLING_FPS, value));
}

function normalizedFrameCap(value: number): number {
  if (!Number.isFinite(value)) return 18;
  return Math.min(MAX_FRAME_CAP, Math.max(MIN_FRAME_CAP, Math.floor(value)));
}

function phaseLabel(phase: RunPhase): string {
  switch (phase) {
    case "queued":
      return "Queued";
    case "sampling":
      return "Sampling";
    case "reconstructing":
      return "Rust/WASM";
    case "done":
      return "Ready";
    case "error":
      return "Failed";
  }
}

function registrationStatusLabel(state: FrameCameraState | null): string {
  if (!state) return "awaiting reconstruction";
  switch (state.registration_status) {
    case "seed":
      return "seed camera";
    case "registered":
      return state.recovered_from_revisit ? "registered · revisit" : "registered";
    case "insufficient_correspondences":
      return `insufficient correspondences · ${state.correspondences}/8`;
    case "pnp_rejected":
      return `PnP rejected · ${state.correspondences} correspondences`;
    case "low_parallax":
      return "low parallax";
    case "not_selected":
      return state.camera_kind === "approximate_motion" ? "approximate motion" : "not selected";
  }
}

function selectedFrameEvidence(state: FrameCameraState | null): string[] {
  if (!state) return [];
  const details = [registrationStatusLabel(state)];
  if (state.correspondences > 0) {
    details.push(`${state.correspondences} correspondences`);
  }
  if (state.inliers > 0) {
    details.push(`${state.inliers} inliers`);
  }
  if (state.median_reprojection_error_pixels !== null) {
    details.push(`${state.median_reprojection_error_pixels.toFixed(2)} px median reprojection`);
  }
  if (state.camera_kind === "seed" || state.camera_kind === "registered") {
    if (state.bundle_adjustment === "accepted") {
      details.push("bundle adjustment accepted");
    } else if (state.bundle_adjustment === "rejected") {
      details.push("bundle adjustment rejected; previous geometry retained");
    }
  }
  if (state.dense_role) {
    details.push(`dense ${state.dense_role}`);
  } else if (state.dense_ineligibility_reason) {
    details.push(`dense-ineligible: ${state.dense_ineligibility_reason}`);
  }
  return details;
}

export default function Home() {
  const [runs, setRuns] = useState<VideoRun[]>([]);
  const [activeRunId, setActiveRunId] = useState("");
  const [batchRunning, setBatchRunning] = useState(false);
  const [samplingFps, setSamplingFps] = useState(1.25);
  const [frameCap, setFrameCap] = useState(18);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number | null>(null);

  function updateRun(id: string, update: Partial<VideoRun>) {
    setRuns((current) =>
      current.map((run) => (run.id === id ? { ...run, ...update } : run)),
    );
  }

  async function runFiles(files: File[]) {
    const runSamplingFps = normalizedSamplingFps(samplingFps);
    const runFrameCap = normalizedFrameCap(frameCap);
    setSamplingFps(runSamplingFps);
    setFrameCap(runFrameCap);

    const nextRuns: VideoRun[] = files.map((file, index) => ({
      id: `${file.name}:${file.size}:${file.lastModified}:${index}`,
      fileName: file.name,
      phase: "queued",
      frames: [],
      reconstruction: null,
      error: "",
      samplingFps: runSamplingFps,
      frameCap: runFrameCap,
    }));

    setRuns(nextRuns);
    setActiveRunId(nextRuns[0].id);
    setSelectedFrameIndex(null);

    for (const [index, file] of files.entries()) {
      const run = nextRuns[index];
      try {
        updateRun(run.id, { phase: "sampling", error: "" });
        const sampled = await sampleVideo(file, {
          framesPerSecond: run.samplingFps,
          maxFrames: run.frameCap,
        });
        updateRun(run.id, {
          phase: "reconstructing",
          frames: previewFrames(sampled),
        });
        const result = await reconstructFrames(sampled);
        updateRun(run.id, { phase: "done", reconstruction: result });
      } catch (caught) {
        updateRun(run.id, {
          phase: "error",
          error: caught instanceof Error ? caught.message : String(caught),
        });
      }
    }
  }

  async function handleVideos(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length === 0 || batchRunning) return;

    setBatchRunning(true);
    try {
      await runFiles(files);
    } finally {
      setBatchRunning(false);
    }
  }

  async function handleTreviDemo() {
    if (batchRunning) return;

    setBatchRunning(true);
    try {
      const response = await fetch(TREVI_DEMO_URL);
      if (!response.ok) {
        throw new Error(`Trevi demo download failed with HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const file = new File([blob], TREVI_DEMO_FILE_NAME, {
        type: blob.type || "video/webm",
        lastModified: 0,
      });
      await runFiles([file]);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      const runSamplingFps = normalizedSamplingFps(samplingFps);
      const runFrameCap = normalizedFrameCap(frameCap);
      const failedRun: VideoRun = {
        id: "trevi-fountain-demo:error",
        fileName: TREVI_DEMO_FILE_NAME,
        phase: "error",
        frames: [],
        reconstruction: null,
        error: message,
        samplingFps: runSamplingFps,
        frameCap: runFrameCap,
      };
      setRuns([failedRun]);
      setActiveRunId(failedRun.id);
      setSelectedFrameIndex(null);
    } finally {
      setBatchRunning(false);
    }
  }

  const activeRun = runs.find((run) => run.id === activeRunId) ?? runs[0] ?? null;
  const processingRun = runs.find(
    (run) => run.phase === "sampling" || run.phase === "reconstructing",
  );
  const processingIndex = processingRun
    ? runs.findIndex((run) => run.id === processingRun.id)
    : -1;
  const failedCount = runs.filter((run) => run.phase === "error").length;
  const readyCount = runs.filter((run) => run.phase === "done").length;
  const statusPhase: StatusPhase = processingRun?.phase ?? activeRun?.phase ?? "idle";
  const frames = activeRun?.frames ?? [];
  const reconstruction = activeRun?.reconstruction ?? null;
  const error = activeRun?.error ?? "";
  const selectedFrame =
    selectedFrameIndex === null ? null : (frames[selectedFrameIndex] ?? null);
  const selectedFrameState =
    selectedFrameIndex === null
      ? null
      : (reconstruction?.camera_state.frames[selectedFrameIndex] ?? null);
  const hasRegisteredGeometry =
    reconstruction?.camera_state.calibrated_seed_cameras.length === 2;
  const selectedIsKeyframe =
    selectedFrameIndex !== null &&
    Boolean(reconstruction?.multi_view.keyframes.includes(selectedFrameIndex));
  const selectedEvidence = selectedFrameEvidence(selectedFrameState);

  const status = processingRun
    ? processingRun.phase === "sampling"
      ? `Sampling video ${processingIndex + 1} of ${runs.length} locally…`
      : `Running Rust/WASM reconstruction for video ${processingIndex + 1} of ${runs.length}…`
    : runs.length > 0
      ? failedCount > 0
        ? `${readyCount} ready · ${failedCount} failed`
        : `${readyCount} ${readyCount === 1 ? "video" : "videos"} ready`
      : "Choose one or more videos to begin";

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">Local browser reconstruction</p>
          <h1>Video to 3D</h1>
          <p className="lede">
            Sample a moving-camera video, reconstruct geometry in Rust/WASM, and inspect why each
            source frame did or did not become an accepted camera.
          </p>
        </div>
        <div className="run-controls">
          <div className="sampling-controls" aria-label="Sampling controls">
            <label>
              <span>Sample FPS</span>
              <input
                type="number"
                min={MIN_SAMPLING_FPS}
                max={MAX_SAMPLING_FPS}
                step="0.25"
                value={samplingFps}
                disabled={batchRunning}
                onChange={(event) => {
                  if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                    setSamplingFps(event.currentTarget.valueAsNumber);
                  }
                }}
              />
            </label>
            <label>
              <span>Frame cap</span>
              <input
                type="number"
                min={MIN_FRAME_CAP}
                max={MAX_FRAME_CAP}
                step="1"
                value={frameCap}
                disabled={batchRunning}
                onChange={(event) => {
                  if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                    setFrameCap(event.currentTarget.valueAsNumber);
                  }
                }}
              />
            </label>
          </div>
          <div className="input-actions">
            <label className="upload-button">
              <input
                type="file"
                accept="video/*"
                multiple
                onChange={handleVideos}
                disabled={batchRunning}
              />
              Select video files
            </label>
            <button
              type="button"
              className="demo-button"
              onClick={() => void handleTreviDemo()}
              disabled={batchRunning}
            >
              Try Trevi test video
            </button>
          </div>
          <small>Sampling settings apply to newly selected videos and the test clip.</small>
          <small className="demo-attribution">
            Trevi Fountain demo:{" "}
            <a href={TREVI_DEMO_FILE_PAGE} target="_blank" rel="noreferrer">
              Benoit-caen / Wikimedia Commons
            </a>{" "}
            ·{" "}
            <a href={TREVI_DEMO_LICENSE} target="_blank" rel="noreferrer">
              CC BY-SA 4.0
            </a>{" "}
            · 18 s · 14.9 MB. The clip is fetched only when you choose it; reconstruction still runs
            locally in your browser.
          </small>
        </div>
      </header>

      <section className="scene-status">
        <strong>Scene detection:</strong> not active yet. The Pages demo currently treats each file as
        one sequence; it does not call scenedetect-rs.
      </section>

      <section className="status-line" aria-live="polite">
        <span className={`status-dot status-${statusPhase}`} />
        <strong>{status}</strong>
        {processingRun?.fileName ?? activeRun?.fileName ? (
          <span>{processingRun?.fileName ?? activeRun?.fileName}</span>
        ) : null}
      </section>

      {runs.length > 0 ? (
        <nav className="video-runs" aria-label="Selected videos">
          {runs.map((run) => (
            <button
              key={run.id}
              type="button"
              className={`video-run${run.id === activeRun?.id ? " video-run-active" : ""}`}
              aria-pressed={run.id === activeRun?.id}
              onClick={() => {
                setActiveRunId(run.id);
                setSelectedFrameIndex(null);
              }}
            >
              <span>{run.fileName}</span>
              <small>{phaseLabel(run.phase)}</small>
            </button>
          ))}
        </nav>
      ) : null}

      {error ? <section className="notice notice-error">{error}</section> : null}

      <section className="workspace">
        <div className="viewer-panel">
          {reconstruction ? (
            <>
              <SceneCanvas
                reconstruction={reconstruction}
                selectedFrameIndex={selectedFrameIndex}
                onSelectFrame={setSelectedFrameIndex}
              />
              {selectedFrame ? (
                <div className="selected-frame-preview">
                  <img
                    src={selectedFrame.thumbnail}
                    alt={`Selected sampled frame ${selectedFrameIndex! + 1}`}
                  />
                  <span>
                    Frame {selectedFrameIndex! + 1} · {selectedFrame.time.toFixed(2)}s
                  </span>
                </div>
              ) : null}
              <div className="viewer-caption">
                Drag to orbit · wheel to zoom · click a camera marker to select its source frame
              </div>
            </>
          ) : (
            <div className="empty-viewer">
              <div className="axis-mark" aria-hidden="true">
                XYZ
              </div>
              <p>The camera track and reconstructed geometry will appear here.</p>
            </div>
          )}
        </div>

        <aside className="method-panel">
          <h2>What happens</h2>
          <ol>
            <li>Sample the video locally at your requested rate, bounded by the frame cap.</li>
            <li>Match local features and build multi-frame tracks in Rust/WASM.</li>
            <li>Register supported cameras, triangulate sparse points, and refine accepted geometry.</li>
          </ol>
          <p className="method-note">
            Squares are accepted camera geometry: blue-center squares are the calibrated seed pair and
            plain squares are additional PnP or revisit registrations. Before calibration, dim circles
            are only approximate motion samples. Dense-eligible cameras are tracked as a separate
            subset and never inferred from the display path.
          </p>
        </aside>
      </section>

      {frames.length > 0 ? (
        <section className="section-block">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Input evidence</p>
              <h2>Sampled frames</h2>
            </div>
            <p>
              {frames.length} frames · {activeRun?.samplingFps.toFixed(2)} target FPS · cap{" "}
              {activeRun?.frameCap}
            </p>
          </div>
          <div className="frame-strip">
            {frames.map((frame, index) => {
              const frameState = reconstruction?.camera_state.frames[index] ?? null;
              const frameStatus = registrationStatusLabel(frameState);
              return (
                <button
                  key={`${frame.time}-${index}`}
                  type="button"
                  className={`frame-card${selectedFrameIndex === index ? " frame-card-selected" : ""}`}
                  aria-pressed={selectedFrameIndex === index}
                  aria-label={`Frame ${index + 1}, ${frame.time.toFixed(2)} seconds, ${frameStatus}`}
                  onClick={() => setSelectedFrameIndex(index)}
                >
                  <img src={frame.thumbnail} alt={`Sampled frame ${index + 1}`} />
                  <span>Frame {index + 1}</span>
                  <small>{frame.time.toFixed(2)}s</small>
                  <small className="frame-card-status">{frameStatus}</small>
                </button>
              );
            })}
          </div>
          {selectedFrameIndex !== null ? (
            <div className="selection-summary">
              <strong>
                Frame {selectedFrameIndex + 1}
                {selectedIsKeyframe ? " · keyframe" : ""}
              </strong>
              {selectedEvidence.length > 0 ? <span>{selectedEvidence.join(" · ")}</span> : null}
            </div>
          ) : null}
        </section>
      ) : null}

      {reconstruction ? (
        <details className="section-block diagnostics">
          <summary>Technical reconstruction evidence</summary>
          <div className="diagnostics-body">
            <div className="table-wrap">
              <table>
                <tbody>
                  <tr>
                    <th>Selected keyframes</th>
                    <td>
                      {reconstruction.multi_view.keyframes.length > 0
                        ? reconstruction.multi_view.keyframes
                            .map((frameIndex) => frameIndex + 1)
                            .join(", ")
                        : "None"}
                    </td>
                  </tr>
                  <tr>
                    <th>Approximate motion samples</th>
                    <td>{reconstruction.camera_state.approximate_motion_samples.length}</td>
                  </tr>
                  <tr>
                    <th>Calibrated seed cameras</th>
                    <td>{reconstruction.camera_state.calibrated_seed_cameras.length}</td>
                  </tr>
                  <tr>
                    <th>Additional registered cameras</th>
                    <td>{reconstruction.camera_state.registered_cameras.length}</td>
                  </tr>
                  <tr>
                    <th>Dense-eligible cameras</th>
                    <td>{reconstruction.camera_state.dense_eligible_cameras.length}</td>
                  </tr>
                  <tr>
                    <th>Camera-track display</th>
                    <td>
                      {hasRegisteredGeometry
                        ? "Accepted seed and registered camera geometry"
                        : `${reconstruction.camera_state.approximate_motion_samples.length} approximate motion samples (not registered)`}
                    </td>
                  </tr>
                  <tr>
                    <th>Feature tracks</th>
                    <td>{reconstruction.multi_view.track_count}</td>
                  </tr>
                  <tr>
                    <th>Sparse points</th>
                    <td>{reconstruction.points.length}</td>
                  </tr>
                  <tr>
                    <th>Coarse dense points</th>
                    <td>{reconstruction.dense_points.length}</td>
                  </tr>
                  <tr>
                    <th>Mesh triangles</th>
                    <td>{reconstruction.mesh_triangles.length}</td>
                  </tr>
                  <tr>
                    <th>Seed pair</th>
                    <td>
                      {reconstruction.calibrated_pair
                        ? `Frame ${reconstruction.calibrated_pair.from_frame + 1} → Frame ${reconstruction.calibrated_pair.to_frame + 1}`
                        : "No calibrated seed pair"}
                    </td>
                  </tr>
                  <tr>
                    <th>Dense camera set</th>
                    <td>
                      {reconstruction.dense.attempted && reconstruction.dense.reference_frame !== null
                        ? `Reference frame ${reconstruction.dense.reference_frame + 1}; source frames ${
                            reconstruction.dense.source_frames.length > 0
                              ? reconstruction.dense.source_frames
                                  .map((frameIndex) => frameIndex + 1)
                                  .join(", ")
                              : "none"
                          }`
                        : reconstruction.dense.skip_reason ?? "Not run"}
                    </td>
                  </tr>
                  <tr>
                    <th>Bundle adjustment</th>
                    <td>
                      {reconstruction.multi_view.bundle_adjustment.accepted
                        ? "Accepted"
                        : reconstruction.multi_view.bundle_adjustment.attempted
                          ? "Rejected; previous geometry retained"
                          : "Not run"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            {reconstruction.warnings.length > 0 ? (
              <ul className="warnings">
                {reconstruction.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            ) : null}
          </div>
        </details>
      ) : null}

      <footer>
        User-selected videos never leave the browser. The optional Trevi test clip is fetched from
        Wikimedia Commons, then processed locally; the GitHub Pages demo has no upload endpoint.
      </footer>
    </main>
  );
}
