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
        ? "Running Rust/WASM reconstruction…"
        : phase === "done"
          ? reconstruction?.calibrated_pair
            ? "Calibrated two-view reconstruction ready"
            : "Conservative sparse preview ready"
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
            the footage. Slice 2 now fits calibrated two-view geometry for the strongest adjacent
            frame pair, recovers relative camera pose, and triangulates real 3D points in Rust/WASM.
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
          <h2>Slice 2 method</h2>
          <ol>
            <li>Sample up to 18 reduced-resolution video frames in the browser.</li>
            <li>Detect and match local image features in Rust/WASM.</li>
            <li>Fit an essential matrix with deterministic RANSAC using estimated pinhole intrinsics.</li>
            <li>Recover relative rotation and translation by cheirality, then triangulate the best pair.</li>
          </ol>
          <p className="method-note">
            Scale is still arbitrary and focal length is estimated from the analysis image unless a
            caller supplies it. Pure-rotation solutions are rejected with a rotation-only fit and
            triangulation-angle gate. Multi-view tracks, PnP, and bundle adjustment remain slice 3.
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
                  Frame {reconstruction.calibrated_pair.from_frame + 1} → {reconstruction.calibrated_pair.to_frame + 1}
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
                        {reconstruction.calibrated_pair.inliers} / {reconstruction.calibrated_pair.matches}
                      </td>
                      <td>{(reconstruction.calibrated_pair.inlier_ratio * 100).toFixed(0)}%</td>
                      <td>{reconstruction.calibrated_pair.focal_pixels.toFixed(1)} px</td>
                      <td>{reconstruction.calibrated_pair.median_sampson_error_pixels.toFixed(2)} px</td>
                      <td>{reconstruction.calibrated_pair.median_reprojection_error_pixels.toFixed(2)} px</td>
                      <td>{reconstruction.calibrated_pair.median_triangulation_angle_degrees.toFixed(2)}°</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}

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
                    <th>Median motion</th>
                    <th>Parallax residual</th>
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
                      <td>{pair.median_motion.toFixed(2)} px</td>
                      <td>{pair.median_parallax_residual.toFixed(2)} px</td>
                      <td>{pair.low_parallax ? "Rejected early" : "RANSAC candidate"}</td>
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
