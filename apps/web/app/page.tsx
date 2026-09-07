"use client";

import { type ChangeEvent, useState } from "react";
import {
  reconstructFrames,
  type ReconstructionResult,
  type SampledFrame,
} from "../src/reconstruction";
import { SceneCanvas } from "../src/SceneCanvas";
import { sampleVideo } from "../src/video";

type Phase = "idle" | "sampling" | "reconstructing" | "done" | "error";

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [fileName, setFileName] = useState("");
  const [frames, setFrames] = useState<SampledFrame[]>([]);
  const [reconstruction, setReconstruction] = useState<ReconstructionResult | null>(null);
  const [error, setError] = useState("");

  async function handleVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setFrames([]);
    setReconstruction(null);
    setError("");

    try {
      setPhase("sampling");
      const sampled = await sampleVideo(file);
      setFrames(sampled);
      setPhase("reconstructing");
      const result = await reconstructFrames(sampled);
      setReconstruction(result);
      setPhase("done");
    } catch (caught) {
      setPhase("error");
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      event.target.value = "";
    }
  }

  const status =
    phase === "sampling"
      ? "Sampling video frames locally…"
      : phase === "reconstructing"
        ? "Running Rust/WASM sparse reconstruction…"
        : phase === "done"
          ? "Sparse preview ready"
          : phase === "error"
            ? "Reconstruction failed"
            : "Choose a video to begin";

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">Rust · WebAssembly · Tauri</p>
          <h1>Video to 3D</h1>
          <p className="lede">
            Turn a moving-camera video into an inspectable sparse 3D reconstruction without uploading
            the footage. This MVP proves frame sampling, Rust-owned feature matching, a camera path,
            and a colored point cloud before the project graduates to calibrated SfM.
          </p>
        </div>
        <label className="upload-button">
          <input
            type="file"
            accept="video/*"
            onChange={handleVideo}
            disabled={phase === "sampling" || phase === "reconstructing"}
          />
          Select video
        </label>
      </header>

      <section className="status-line" aria-live="polite">
        <span className={`status-dot status-${phase}`} />
        <strong>{status}</strong>
        {fileName ? <span>{fileName}</span> : null}
      </section>

      {error ? <section className="notice notice-error">{error}</section> : null}

      <section className="workspace">
        <div className="viewer-panel">
          {reconstruction ? (
            <>
              <SceneCanvas reconstruction={reconstruction} />
              <div className="viewer-caption">
                Drag to orbit · wheel to zoom · squares are sampled camera positions
              </div>
            </>
          ) : (
            <div className="empty-viewer">
              <div className="axis-mark" aria-hidden="true">
                XYZ
              </div>
              <p>The reconstructed camera path and sparse colored geometry will appear here.</p>
              <p>Best first test: slowly move sideways around a textured static object.</p>
            </div>
          )}
        </div>

        <aside className="method-panel">
          <h2>MVP method</h2>
          <ol>
            <li>Sample up to 18 reduced-resolution video frames in the browser.</li>
            <li>Detect corners and normalized patch descriptors in Rust/WASM.</li>
            <li>Match adjacent frames and measure image-space parallax.</li>
            <li>Estimate an approximate camera path and sparse depth preview.</li>
          </ol>
          <p className="method-note">
            This slice is deliberately uncalibrated. Metric scale and camera rotation are not yet
            recovered; the next SfM slice adds essential-matrix RANSAC, triangulation, tracks, and
            bundle adjustment.
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
                <p className="eyebrow">Algorithm evidence</p>
                <h2>Adjacent-frame matching</h2>
              </div>
              <p>
                {reconstruction.points.length} sparse point observations · {reconstruction.cameras.length}{" "}
                camera samples
              </p>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Pair</th>
                    <th>Features</th>
                    <th>Matches</th>
                    <th>Median motion</th>
                    <th>Assessment</th>
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
                      <td>{pair.median_motion.toFixed(2)} px</td>
                      <td>{pair.low_parallax ? "Weak parallax" : "Usable"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}

      <footer>All video processing is local. The static site has no upload endpoint.</footer>
    </main>
  );
}
