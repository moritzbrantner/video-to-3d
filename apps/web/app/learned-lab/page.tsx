"use client";

import { type ChangeEvent, useState } from "react";
import { SceneCanvas } from "../../src/SceneCanvas";
import {
  LEARNED_PROVIDER_CATALOG,
  benchmarkLearnedMode,
  learnedModeLabel,
  type LearnedBenchmarkSuite,
  type LearnedReconstructionMode,
} from "../../src/learnedDepth";
import { reconstructFrames, type ReconstructionResult } from "../../src/reconstruction";
import { sampleVideo } from "../../src/video";

type LabPhase = "idle" | "sampling" | "reconstructing" | "learned" | "done" | "error";

function formatBytes(bytes: number): string {
  return bytes >= 1_000_000 ? `${(bytes / 1_000_000).toFixed(0)} MB` : `${bytes} B`;
}

function formatMilliseconds(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(0)} ms`;
}

function formatPercent(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function phaseText(phase: LabPhase): string {
  switch (phase) {
    case "idle":
      return "Choose a video to benchmark browser-local learned reconstruction providers.";
    case "sampling":
      return "Sampling video locally…";
    case "reconstructing":
      return "Building the authoritative Rust/WASM camera reconstruction…";
    case "learned":
      return "Running learned inference sequentially on accepted registered frames…";
    case "done":
      return "Benchmark complete.";
    case "error":
      return "Benchmark failed.";
  }
}

export default function LearnedLab() {
  const [mode, setMode] = useState<LearnedReconstructionMode>("benchmark");
  const [phase, setPhase] = useState<LabPhase>("idle");
  const [fileName, setFileName] = useState("");
  const [frameCount, setFrameCount] = useState(0);
  const [reconstruction, setReconstruction] = useState<ReconstructionResult | null>(null);
  const [benchmark, setBenchmark] = useState<LearnedBenchmarkSuite | null>(null);
  const [error, setError] = useState("");

  const running = phase === "sampling" || phase === "reconstructing" || phase === "learned";

  async function handleVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || running) return;

    const runMode = mode;
    setFileName(file.name);
    setFrameCount(0);
    setReconstruction(null);
    setBenchmark(null);
    setError("");

    try {
      setPhase("sampling");
      const frames = await sampleVideo(file, { framesPerSecond: 1.25, maxFrames: 18 });
      setFrameCount(frames.length);

      setPhase("reconstructing");
      const classic = await reconstructFrames(frames);
      setReconstruction(classic);

      if (runMode !== "classic") {
        setPhase("learned");
        setBenchmark(await benchmarkLearnedMode(runMode, frames, classic));
      }
      setPhase("done");
    } catch (caught) {
      setPhase("error");
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">Browser model evidence</p>
          <h1>Learned reconstruction lab</h1>
          <p className="lede">
            Compare small browser-local depth and geometry models on the exact frames accepted by the
            Rust camera reconstruction. Learned output remains evidence until it crosses the shared
            Rust validation and fusion boundary.
          </p>
        </div>
        <div className="run-controls">
          <label style={{ display: "grid", gap: 6, color: "var(--muted)", fontSize: "0.78rem" }}>
            <span>Learned model</span>
            <select
              value={mode}
              disabled={running}
              onChange={(event) => setMode(event.currentTarget.value as LearnedReconstructionMode)}
              style={{
                minHeight: 42,
                minWidth: 270,
                border: "1px solid var(--line)",
                borderRadius: 9,
                padding: "0 10px",
                background: "var(--panel)",
                color: "var(--text)",
              }}
            >
              <option value="classic">Classic Rust/WASM only</option>
              {LEARNED_PROVIDER_CATALOG.map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {provider.label}
                </option>
              ))}
              <option value="benchmark">Benchmark both learned providers</option>
            </select>
          </label>
          <label className="upload-button">
            <input type="file" accept="video/*" disabled={running} onChange={handleVideo} />
            {running ? "Running…" : "Select benchmark video"}
          </label>
          <small>Selected mode: {learnedModeLabel(mode)}</small>
        </div>
      </header>

      <section className="status-line" aria-live="polite">
        <span className={`status-dot status-${phase === "learned" ? "reconstructing" : phase}`} />
        <strong>{phaseText(phase)}</strong>
        {fileName ? <span>{fileName}</span> : null}
      </section>

      {error ? <section className="notice notice-error">{error}</section> : null}

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Execution contract</p>
            <h2>What this lab measures</h2>
          </div>
          <p>{frameCount > 0 ? `${frameCount} sampled frames` : "No video loaded"}</p>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 12,
          }}
        >
          {LEARNED_PROVIDER_CATALOG.map((provider) => (
            <article
              key={provider.id}
              style={{ border: "1px solid var(--line)", borderRadius: 12, padding: 14 }}
            >
              <strong>{provider.label}</strong>
              <p style={{ color: "var(--muted)", lineHeight: 1.5, fontSize: "0.82rem" }}>
                {provider.description}
              </p>
              <small style={{ color: "var(--muted)" }}>
                {provider.runtime} · ~{formatBytes(provider.approximateDownloadBytes)} ·{" "}
                {provider.modelReference}
              </small>
            </article>
          ))}
        </div>
      </section>

      {benchmark ? (
        <section className="section-block">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Measured on this device</p>
              <h2>Learned provider benchmark</h2>
            </div>
            <p>
              frames{" "}
              {benchmark.selectedFrames.length > 0
                ? benchmark.selectedFrames.map((frame) => frame + 1).join(", ")
                : "none"}
            </p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>Status</th>
                  <th>Backend</th>
                  <th>Load / compile</th>
                  <th>Median frame</th>
                  <th>P90 frame</th>
                  <th>Finite evidence</th>
                  <th>Confidence &gt; 0.5</th>
                  <th>Geometry usable</th>
                  <th>Geometry median error</th>
                  <th>Geometry agreement ≤15%</th>
                </tr>
              </thead>
              <tbody>
                {benchmark.providers.map((provider) => (
                  <tr key={provider.providerId}>
                    <td>{provider.label}</td>
                    <td>{provider.status}</td>
                    <td>{provider.runtimeLabel ?? provider.backend ?? "—"}</td>
                    <td>{formatMilliseconds(provider.loadMs)}</td>
                    <td>{formatMilliseconds(provider.medianInferenceMs)}</td>
                    <td>{formatMilliseconds(provider.p90InferenceMs)}</td>
                    <td>{formatPercent(provider.finiteEvidenceRatio)}</td>
                    <td>{formatPercent(provider.confidenceCoverage)}</td>
                    <td>
                      {provider.geometricAgreement
                        ? `${provider.geometricAgreement.usableFrames}/${provider.geometricAgreement.evaluatedFrames}`
                        : "—"}
                    </td>
                    <td>{formatPercent(provider.geometricAgreement?.medianRelativeError ?? null)}</td>
                    <td>
                      {formatPercent(
                        provider.geometricAgreement?.meanAgreementRatio15Percent ?? null,
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="warnings">
            {benchmark.providers.map((provider) => (
              <li key={`${provider.providerId}-diagnostic`}>
                <strong>{provider.label}:</strong> {provider.diagnostic}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {reconstruction ? (
        <section className="workspace">
          <div className="viewer-panel">
            <SceneCanvas
              reconstruction={reconstruction}
              selectedFrameIndex={null}
              onSelectFrame={() => undefined}
            />
            <div className="viewer-caption">
              Classical geometry stays authoritative while learned providers are evaluated.
            </div>
          </div>
          <aside className="method-panel">
            <h2>Classical baseline</h2>
            <ol>
              <li>
                {reconstruction.camera_state.calibrated_seed_cameras.length} calibrated seed cameras.
              </li>
              <li>
                {reconstruction.camera_state.registered_cameras.length} additional registered cameras.
              </li>
              <li>{reconstruction.dense_points.length} accepted dense points.</li>
              <li>{reconstruction.mesh_triangles.length} accepted mesh triangles.</li>
            </ol>
            <p className="method-note">
              Learned inference is deliberately restricted to accepted registered frames and runs one
              frame at a time so browser GPU memory does not scale with the full video batch.
            </p>
          </aside>
        </section>
      ) : null}

      <footer>
        Video pixels stay local. Model weights are fetched into the browser and may be cached by the
        browser runtime; no reconstruction upload endpoint is used.
      </footer>
    </main>
  );
}
