"use client";

import { type ChangeEvent, useState } from "react";
import {
  reconstructFrames,
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
};

function previewFrames(frames: SampledFrame[]): PreviewFrame[] {
  return frames.map((frame) => ({
    width: frame.width,
    height: frame.height,
    thumbnail: frame.thumbnail,
    time: frame.time,
  }));
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

export default function Home() {
  const [runs, setRuns] = useState<VideoRun[]>([]);
  const [activeRunId, setActiveRunId] = useState("");
  const [batchRunning, setBatchRunning] = useState(false);

  function updateRun(id: string, update: Partial<VideoRun>) {
    setRuns((current) =>
      current.map((run) => (run.id === id ? { ...run, ...update } : run)),
    );
  }

  async function handleVideos(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length === 0 || batchRunning) return;

    const nextRuns: VideoRun[] = files.map((file, index) => ({
      id: `${file.name}:${file.size}:${file.lastModified}:${index}`,
      fileName: file.name,
      phase: "queued",
      frames: [],
      reconstruction: null,
      error: "",
    }));

    setRuns(nextRuns);
    setActiveRunId(nextRuns[0].id);
    setBatchRunning(true);

    for (const [index, file] of files.entries()) {
      const runId = nextRuns[index].id;
      try {
        updateRun(runId, { phase: "sampling", error: "" });
        const sampled = await sampleVideo(file);
        updateRun(runId, {
          phase: "reconstructing",
          frames: previewFrames(sampled),
        });
        const result = await reconstructFrames(sampled);
        updateRun(runId, { phase: "done", reconstruction: result });
      } catch (caught) {
        updateRun(runId, {
          phase: "error",
          error: caught instanceof Error ? caught.message : String(caught),
        });
      }
    }

    setBatchRunning(false);
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
  const status = processingRun
    ? processingRun.phase === "sampling"
      ? `Sampling video ${processingIndex + 1} of ${runs.length} locally…`
      : `Running Rust/WASM reconstruction for video ${processingIndex + 1} of ${runs.length}…`
    : runs.length > 0
      ? failedCount > 0
        ? `${readyCount} ready · ${failedCount} failed`
        : runs.length === 1 && reconstruction?.calibrated_pair
          ? "Calibrated two-view reconstruction and multi-view track graph ready"
          : `${readyCount} ${readyCount === 1 ? "video" : "videos"} ready`
      : "Choose one or more videos to begin";

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">Next.js static export · Rust · WebAssembly · Tauri</p>
          <h1>Video to 3D</h1>
          <p className="lede">
            Turn moving-camera video into an inspectable sparse 3D reconstruction without uploading
            the footage. Rust/WASM now chains adjacent matches into multi-frame tracks and screens
            keyframe candidates by overlap and parallax. The displayed geometry still comes from the
            strongest calibrated adjacent pair until incremental registration is implemented.
          </p>
        </div>
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
      </header>

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
              onClick={() => setActiveRunId(run.id)}
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
              <SceneCanvas reconstruction={reconstruction} />
              <div className="viewer-caption">
                Drag to orbit · wheel to zoom · squares are registered or sampled camera positions
              </div>
            </>
          ) : (
            <div className="empty-viewer">
              <div className="axis-mark" aria-hidden="true">
                XYZ
              </div>
              <p>The reconstructed cameras and sparse colored geometry will appear here.</p>
              <p>Best first test: slowly move sideways around a textured static object.</p>
            </div>
          )}
        </div>

        <aside className="method-panel">
          <h2>Slice 3 foundation</h2>
          <ol>
            <li>Decode and sample up to 18 reduced-resolution frames per video in the browser.</li>
            <li>Detect and match local image features in Rust/WASM.</li>
            <li>Chain one-to-one adjacent matches into deterministic multi-frame feature tracks.</li>
            <li>Select keyframe candidates from track overlap and accumulated residual parallax.</li>
          </ol>
          <p className="method-note">
            Selected videos remain independent and local. The existing calibrated two-view result still
            owns displayed camera pose and triangulated geometry. Incremental camera registration, PnP,
            bundle adjustment, loop handling, and cross-video tracks remain later work in slice 3 or 6.
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
              {frames.length} local keyframe candidates at {frames[0].width} × {frames[0].height}
            </p>
          </div>
          <div className="frame-strip">
            {frames.map((frame, index) => (
              <figure key={`${frame.time}-${index}`}>
                <img src={frame.thumbnail} alt={`Sampled frame ${index + 1}`} />
                <figcaption>{frame.time.toFixed(2)}s</figcaption>
              </figure>
            ))}
          </div>
        </section>
      ) : null}

      {reconstruction ? (
        <>
          {reconstruction.calibrated_pair ? (
            <section className="section-block">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Calibrated geometry evidence</p>
                  <h2>Selected two-view pair</h2>
                </div>
                <p>
                  Frame {reconstruction.calibrated_pair.from_frame + 1} →{" "}
                  {reconstruction.calibrated_pair.to_frame + 1}
                </p>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Inliers</th>
                      <th>Inlier ratio</th>
                      <th>Focal estimate</th>
                      <th>Sampson error</th>
                      <th>Reprojection error</th>
                      <th>Triangulation angle</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        {reconstruction.calibrated_pair.inliers} /{" "}
                        {reconstruction.calibrated_pair.matches}
                      </td>
                      <td>{(reconstruction.calibrated_pair.inlier_ratio * 100).toFixed(0)}%</td>
                      <td>{reconstruction.calibrated_pair.focal_pixels.toFixed(1)} px</td>
                      <td>
                        {reconstruction.calibrated_pair.median_sampson_error_pixels.toFixed(2)} px
                      </td>
                      <td>
                        {reconstruction.calibrated_pair.median_reprojection_error_pixels.toFixed(2)} px
                      </td>
                      <td>
                        {reconstruction.calibrated_pair.median_triangulation_angle_degrees.toFixed(2)}°
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}

          <section className="section-block">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Multi-view evidence</p>
                <h2>Feature-track graph and keyframes</h2>
              </div>
              <p>Rust-owned diagnostics; not yet a registered multi-camera reconstruction</p>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Selected keyframes</th>
                    <th>Feature tracks</th>
                    <th>Tracks across 3+ frames</th>
                    <th>Longest track</th>
                    <th>Track observations</th>
                    <th>Linked adjacent pairs</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      {reconstruction.multi_view.keyframes.length > 0
                        ? reconstruction.multi_view.keyframes
                            .map((frameIndex) => frameIndex + 1)
                            .join(", ")
                        : "None"}
                    </td>
                    <td>{reconstruction.multi_view.track_count}</td>
                    <td>{reconstruction.multi_view.tracks_three_plus}</td>
                    <td>{reconstruction.multi_view.longest_track} frames</td>
                    <td>{reconstruction.multi_view.observations}</td>
                    <td>
                      {reconstruction.multi_view.linked_pairs} / {reconstruction.pairs.length}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {reconstruction.warnings.length > 0 ? (
            <section className="section-block">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Interpretation</p>
                  <h2>Reconstruction notes</h2>
                </div>
              </div>
              <ul className="warnings">
                {reconstruction.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </section>
          ) : null}

          <section className="section-block">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Matching evidence</p>
                <h2>Adjacent-frame screening</h2>
              </div>
              <p>
                {reconstruction.points.length} sparse point observations · {reconstruction.cameras.length}{" "}
                displayed cameras
              </p>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Pair</th>
                    <th>Features</th>
                    <th>Matches</th>
                    <th>Overlap</th>
                    <th>Median motion</th>
                    <th>Parallax residual</th>
                    <th>Keyframe</th>
                    <th>Screening</th>
                  </tr>
                </thead>
                <tbody>
                  {reconstruction.pairs.map((pair) => (
                    <tr key={pair.from_frame}>
                      <td>
                        {pair.from_frame + 1} → {pair.to_frame + 1}
                      </td>
                      <td>
                        {pair.features_from} / {pair.features_to}
                      </td>
                      <td>{pair.matches}</td>
                      <td>{(pair.overlap_ratio * 100).toFixed(0)}%</td>
                      <td>{pair.median_motion.toFixed(2)} px</td>
                      <td>{pair.median_parallax_residual.toFixed(2)} px</td>
                      <td>
                        {reconstruction.multi_view.keyframes.includes(pair.to_frame)
                          ? "Selected"
                          : "Skipped"}
                      </td>
                      <td>{pair.low_parallax ? "Weak adjacent baseline" : "RANSAC candidate"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}

      <footer>All video processing is local. The static GitHub Pages demo has no upload endpoint.</footer>
    </main>
  );
}
