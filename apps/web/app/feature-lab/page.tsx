"use client";

import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
} from "react";

import {
  analyzeFeaturePair,
  defaultFeatureOptions,
  type FeatureAlgorithm,
  type FeatureAnalysis,
  type FeatureOptions,
} from "../../src/featureLab";

const FRAME_WIDTH = 360;
const FRAME_HEIGHT = 240;
const GAP = 28;
const TARGET_X = FRAME_WIDTH + GAP;

type AnalysisByAlgorithm = Partial<Record<FeatureAlgorithm, FeatureAnalysis>>;

function syntheticFrames(): { source: ImageData; target: ImageData } {
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = FRAME_WIDTH;
  sourceCanvas.height = FRAME_HEIGHT;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  if (!sourceContext) throw new Error("Canvas 2D is unavailable.");

  sourceContext.fillStyle = "#e8e5dc";
  sourceContext.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
  sourceContext.fillStyle = "#27313b";
  sourceContext.fillRect(24, 28, 86, 54);
  sourceContext.fillStyle = "#b95f3f";
  sourceContext.fillRect(137, 35, 72, 91);
  sourceContext.fillStyle = "#47745e";
  sourceContext.beginPath();
  sourceContext.arc(281, 68, 36, 0, Math.PI * 2);
  sourceContext.fill();
  sourceContext.strokeStyle = "#15202a";
  sourceContext.lineWidth = 4;
  sourceContext.beginPath();
  sourceContext.moveTo(38, 181);
  sourceContext.lineTo(112, 135);
  sourceContext.lineTo(172, 201);
  sourceContext.lineTo(232, 149);
  sourceContext.lineTo(324, 194);
  sourceContext.stroke();
  sourceContext.font = "bold 39px system-ui, sans-serif";
  sourceContext.fillStyle = "#263746";
  sourceContext.fillText("3D", 246, 166);

  for (let y = 16; y < FRAME_HEIGHT - 12; y += 23) {
    for (let x = 14; x < FRAME_WIDTH - 12; x += 27) {
      const value = (x * 13 + y * 17) % 95;
      sourceContext.fillStyle = `rgb(${70 + value}, ${82 + (value * 3) % 92}, ${92 + (value * 5) % 83})`;
      sourceContext.fillRect(x, y, 4 + ((x + y) % 5), 4 + ((x * y) % 4));
    }
  }

  const targetCanvas = document.createElement("canvas");
  targetCanvas.width = FRAME_WIDTH;
  targetCanvas.height = FRAME_HEIGHT;
  const targetContext = targetCanvas.getContext("2d", { willReadFrequently: true });
  if (!targetContext) throw new Error("Canvas 2D is unavailable.");

  targetContext.fillStyle = "#e8e5dc";
  targetContext.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
  targetContext.save();
  targetContext.translate(FRAME_WIDTH / 2 + 18, FRAME_HEIGHT / 2 - 5);
  targetContext.rotate((7 * Math.PI) / 180);
  targetContext.scale(1.02, 1.02);
  targetContext.drawImage(sourceCanvas, -FRAME_WIDTH / 2, -FRAME_HEIGHT / 2);
  targetContext.restore();
  targetContext.fillStyle = "rgba(255, 255, 255, 0.08)";
  targetContext.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);

  return {
    source: sourceContext.getImageData(0, 0, FRAME_WIDTH, FRAME_HEIGHT),
    target: targetContext.getImageData(0, 0, FRAME_WIDTH, FRAME_HEIGHT),
  };
}

async function decodeImage(file: File): Promise<ImageData> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = FRAME_WIDTH;
  canvas.height = FRAME_HEIGHT;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D is unavailable.");

  context.fillStyle = "#111";
  context.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
  const scale = Math.min(FRAME_WIDTH / bitmap.width, FRAME_HEIGHT / bitmap.height);
  const width = bitmap.width * scale;
  const height = bitmap.height * scale;
  context.drawImage(
    bitmap,
    (FRAME_WIDTH - width) / 2,
    (FRAME_HEIGHT - height) / 2,
    width,
    height,
  );
  bitmap.close();
  return context.getImageData(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[middle];
  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function renderAnalysis(
  canvas: HTMLCanvasElement,
  source: ImageData,
  target: ImageData,
  analysis: FeatureAnalysis | undefined,
): void {
  canvas.width = FRAME_WIDTH * 2 + GAP;
  canvas.height = FRAME_HEIGHT;
  const context = canvas.getContext("2d");
  if (!context) return;

  context.putImageData(source, 0, 0);
  context.putImageData(target, TARGET_X, 0);
  context.fillStyle = "#121820";
  context.fillRect(FRAME_WIDTH, 0, GAP, FRAME_HEIGHT);

  if (!analysis) return;

  context.lineWidth = 1;
  context.strokeStyle = "rgba(255, 220, 95, 0.38)";
  for (const match of analysis.matches) {
    const sourcePoint = analysis.source_features[match.source_index];
    const targetPoint = analysis.target_features[match.target_index];
    if (!sourcePoint || !targetPoint) continue;
    context.beginPath();
    context.moveTo(sourcePoint.x, sourcePoint.y);
    context.lineTo(TARGET_X + targetPoint.x, targetPoint.y);
    context.stroke();
  }

  context.lineWidth = 1.5;
  context.strokeStyle = "#ffe57a";
  for (const point of analysis.source_features) {
    context.beginPath();
    context.arc(point.x, point.y, 2.7, 0, Math.PI * 2);
    context.stroke();
  }
  for (const point of analysis.target_features) {
    context.beginPath();
    context.arc(TARGET_X + point.x, point.y, 2.7, 0, Math.PI * 2);
    context.stroke();
  }
}

export default function FeatureLabPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [source, setSource] = useState<ImageData | null>(null);
  const [target, setTarget] = useState<ImageData | null>(null);
  const [algorithm, setAlgorithm] = useState<FeatureAlgorithm>("orb_style");
  const [options, setOptions] = useState<FeatureOptions>(defaultFeatureOptions);
  const [analyses, setAnalyses] = useState<AnalysisByAlgorithm>({});
  const [status, setStatus] = useState("Preparing the built-in fixture…");

  async function runBoth(
    nextSource: ImageData | null = source,
    nextTarget: ImageData | null = target,
    nextOptions: FeatureOptions = options,
  ): Promise<void> {
    if (!nextSource || !nextTarget) {
      setStatus("Choose both frames first.");
      return;
    }

    setStatus("Running both feature pipelines in WebAssembly…");
    try {
      const [baseline, orb] = await Promise.all([
        analyzeFeaturePair(
          nextSource,
          nextTarget,
          "baseline_harris_patch",
          nextOptions,
        ),
        analyzeFeaturePair(nextSource, nextTarget, "orb_style", nextOptions),
      ]);
      setAnalyses({ baseline_harris_patch: baseline, orb_style: orb });
      setStatus("Ready. Switch pipelines to compare the same frames.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  function loadSynthetic(): void {
    const frames = syntheticFrames();
    setSource(frames.source);
    setTarget(frames.target);
    setAnalyses({});
    void runBoth(frames.source, frames.target);
  }

  useEffect(() => {
    const frames = syntheticFrames();
    setSource(frames.source);
    setTarget(frames.target);
    void runBoth(frames.source, frames.target, defaultFeatureOptions);
    // The fixture should run once on entry; later tuning is explicit via Run both.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedAnalysis = analyses[algorithm];
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !source || !target) return;
    renderAnalysis(canvas, source, target, selectedAnalysis);
  }, [source, target, selectedAnalysis]);

  async function replaceFrame(
    side: "source" | "target",
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const image = await decodeImage(file);
      if (side === "source") setSource(image);
      else setTarget(image);
      setAnalyses({});
      setStatus("Frame changed. Run both pipelines to compare it.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      event.target.value = "";
    }
  }

  const baseline = analyses.baseline_harris_patch;
  const orb = analyses.orb_style;
  const medianDx = selectedAnalysis
    ? median(selectedAnalysis.matches.map((match) => match.dx))
    : null;
  const medianDy = selectedAnalysis
    ? median(selectedAnalysis.matches.map((match) => match.dy))
    : null;

  return (
    <main
      style={{
        maxWidth: 1120,
        margin: "0 auto",
        padding: "32px 24px 56px",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <header style={{ marginBottom: 24 }}>
        <p style={{ margin: "0 0 6px", opacity: 0.68 }}>video-to-3d</p>
        <h1 style={{ margin: "0 0 10px", fontSize: 32 }}>Feature matching lab</h1>
        <p style={{ margin: 0, maxWidth: 850, lineHeight: 1.55 }}>
          Compare the existing small Harris + normalized-patch baseline with the stronger
          multi-scale FAST, orientation and rotated-BRIEF pipeline. Both run against exactly
          the same frames; the reconstruction pipeline is not silently changed by this lab.
        </p>
      </header>

      <section
        style={{
          display: "flex",
          gap: 10,
          flexWrap: "wrap",
          alignItems: "center",
          marginBottom: 18,
        }}
      >
        <button
          type="button"
          onClick={() => setAlgorithm("baseline_harris_patch")}
          aria-pressed={algorithm === "baseline_harris_patch"}
        >
          Simple baseline
        </button>
        <button
          type="button"
          onClick={() => setAlgorithm("orb_style")}
          aria-pressed={algorithm === "orb_style"}
        >
          ORB-style
        </button>
        <button type="button" onClick={() => void runBoth()}>
          Run both
        </button>
        <button type="button" onClick={loadSynthetic}>
          Reset synthetic example
        </button>
      </section>

      <canvas
        ref={canvasRef}
        aria-label="Source and target frames with detected points and accepted matches"
        style={{
          display: "block",
          width: "100%",
          maxWidth: FRAME_WIDTH * 2 + GAP,
          height: "auto",
          border: "1px solid rgba(127, 127, 127, 0.4)",
          marginBottom: 14,
        }}
      />

      <p style={{ margin: "0 0 6px", lineHeight: 1.5 }}>
        <strong>{algorithm === "orb_style" ? "ORB-style" : "Simple baseline"}</strong>
        {selectedAnalysis
          ? ` · ${selectedAnalysis.source_features.length} source points · ${selectedAnalysis.target_features.length} target points · ${selectedAnalysis.matches.length} mutual matches`
          : " · not run"}
        {medianDx !== null && medianDy !== null
          ? ` · median motion ${medianDx.toFixed(1)} px, ${medianDy.toFixed(1)} px`
          : ""}
      </p>
      <p style={{ margin: "0 0 22px", opacity: 0.72, lineHeight: 1.45 }}>{status}</p>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 20, marginBottom: 10 }}>Same-frame comparison</h2>
        <table style={{ borderCollapse: "collapse", minWidth: 520 }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", padding: "5px 16px 5px 0" }}>Pipeline</th>
              <th style={{ textAlign: "right", padding: "5px 12px" }}>Source points</th>
              <th style={{ textAlign: "right", padding: "5px 12px" }}>Target points</th>
              <th style={{ textAlign: "right", padding: "5px 0 5px 12px" }}>Matches</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ padding: "5px 16px 5px 0" }}>Simple baseline</td>
              <td style={{ textAlign: "right", padding: "5px 12px" }}>
                {baseline?.source_features.length ?? "—"}
              </td>
              <td style={{ textAlign: "right", padding: "5px 12px" }}>
                {baseline?.target_features.length ?? "—"}
              </td>
              <td style={{ textAlign: "right", padding: "5px 0 5px 12px" }}>
                {baseline?.matches.length ?? "—"}
              </td>
            </tr>
            <tr>
              <td style={{ padding: "5px 16px 5px 0" }}>ORB-style</td>
              <td style={{ textAlign: "right", padding: "5px 12px" }}>
                {orb?.source_features.length ?? "—"}
              </td>
              <td style={{ textAlign: "right", padding: "5px 12px" }}>
                {orb?.target_features.length ?? "—"}
              </td>
              <td style={{ textAlign: "right", padding: "5px 0 5px 12px" }}>
                {orb?.matches.length ?? "—"}
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(235px, 1fr))",
          gap: 18,
          marginBottom: 26,
        }}
      >
        <label>
          Maximum points: <strong>{options.max_features}</strong>
          <input
            type="range"
            min={80}
            max={800}
            step={40}
            value={options.max_features}
            onChange={(event) =>
              setOptions((current) => ({
                ...current,
                max_features: Number(event.target.value),
              }))
            }
            style={{ display: "block", width: "100%", marginTop: 8 }}
          />
        </label>
        <label>
          Minimum spacing: <strong>{options.min_feature_distance} px</strong>
          <input
            type="range"
            min={3}
            max={14}
            step={1}
            value={options.min_feature_distance}
            onChange={(event) =>
              setOptions((current) => ({
                ...current,
                min_feature_distance: Number(event.target.value),
              }))
            }
            style={{ display: "block", width: "100%", marginTop: 8 }}
          />
        </label>
        <label>
          FAST threshold: <strong>{options.fast_threshold}</strong>
          <input
            type="range"
            min={6}
            max={48}
            step={2}
            value={options.fast_threshold}
            onChange={(event) =>
              setOptions((current) => ({
                ...current,
                fast_threshold: Number(event.target.value),
              }))
            }
            style={{ display: "block", width: "100%", marginTop: 8 }}
          />
        </label>
        <label>
          Ratio threshold: <strong>{options.ratio_threshold.toFixed(2)}</strong>
          <input
            type="range"
            min={0.6}
            max={0.95}
            step={0.01}
            value={options.ratio_threshold}
            onChange={(event) =>
              setOptions((current) => ({
                ...current,
                ratio_threshold: Number(event.target.value),
              }))
            }
            style={{ display: "block", width: "100%", marginTop: 8 }}
          />
        </label>
      </section>

      <section style={{ marginBottom: 22 }}>
        <h2 style={{ fontSize: 20, marginBottom: 10 }}>Try your own pair</h2>
        <p style={{ marginTop: 0, lineHeight: 1.5 }}>
          Images are resampled locally to {FRAME_WIDTH}×{FRAME_HEIGHT}. Nothing is uploaded.
        </p>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
          <label>
            Source frame{" "}
            <input
              type="file"
              accept="image/*"
              onChange={(event) => void replaceFrame("source", event)}
            />
          </label>
          <label>
            Target frame{" "}
            <input
              type="file"
              accept="image/*"
              onChange={(event) => void replaceFrame("target", event)}
            />
          </label>
        </div>
      </section>

      <section style={{ maxWidth: 900, lineHeight: 1.55 }}>
        <h2 style={{ fontSize: 20, marginBottom: 8 }}>What is different?</h2>
        <p style={{ marginTop: 0 }}>
          The baseline uses one-scale Harris corners and mean-centred pixel patches inside a local
          search radius. The ORB-style path builds an image pyramid, finds FAST-9 corners, estimates
          an intensity-centroid orientation, computes 256 rotated BRIEF comparisons, then uses
          Hamming distance with ratio and mutual-match checks. The geometric reconstruction gates
          remain separate, so better-looking feature matches still have to prove themselves through
          calibrated two-view and multi-view evidence before becoming authoritative.
        </p>
      </section>
    </main>
  );
}
