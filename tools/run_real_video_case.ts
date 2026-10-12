import { spawnSync } from "node:child_process";
import { existsSync, lstatSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, join, resolve, sep } from "node:path";
import { buildVideoSamplingPlan } from "../apps/web/src/videoSampling";

type CorpusCase = {
  id: string;
  asset: {
    expected_file: string;
  };
  preparation?: {
    trim: {start_seconds: number; end_seconds: number};
    plan: unknown;
  };
};

type CorpusManifest = {
  sampling: {
    frames_per_second: number;
    max_frames: number;
  };
  cases: CorpusCase[];
};

type ProbeResult = {
  streams?: Array<{ width?: number; height?: number; avg_frame_rate?: string }>;
  format?: { duration?: string };
};

function argument(name: string, fallback?: string): string {
  const index = process.argv.indexOf(name);
  if (index >= 0 && process.argv[index + 1]) return process.argv[index + 1];
  if (fallback !== undefined) return fallback;
  throw new Error(`missing required argument ${name}`);
}

function run(program: string, args: string[], capture = false): string {
  const result = spawnSync(program, args, {
    encoding: "utf8",
    stdio: capture ? ["ignore", "pipe", "pipe"] : "inherit",
  });
  if (result.status !== 0) {
    const detail = capture ? `\n${result.stderr}` : "";
    throw new Error(`${program} exited with status ${result.status}${detail}`);
  }
  return capture ? result.stdout : "";
}

function parseFrameRate(value: string | undefined): number {
  if (!value) return Number.NaN;
  const [numeratorText, denominatorText = "1"] = value.split("/");
  const numerator = Number(numeratorText);
  const denominator = Number(denominatorText);
  return denominator > 0 ? numerator / denominator : Number.NaN;
}

function canonicalPath(path: string): string {
  const tail: string[] = [];
  let existing = path;
  while (!existsSync(existing)) {
    tail.unshift(basename(existing));
    existing = dirname(existing);
  }
  return join(realpathSync(existing), ...tail);
}

function containedPath(root: string, relative: string): string {
  if (isAbsolute(relative) || relative.split(/[\\/]/).includes("..") || relative.includes("\\")) {
    throw new Error("benchmark path must be a safe relative path");
  }
  const base = canonicalPath(root);
  const path = canonicalPath(resolve(base, relative));
  if (!path.startsWith(base + sep)) {
    throw new Error("benchmark path escapes declared root");
  }
  return path;
}

const caseId = argument("--case");
if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(caseId)) { throw new Error("unsafe benchmark case id"); }
const assetRoot = resolve(argument("--asset-root", ".benchmark-data"));
const outputRoot = resolve(argument("--output-root", "target/real-video"));
const manifestPath = resolve(argument("--manifest", "benchmarks/real-video-corpus.json"));
// Approval is selected by immutable case identity, not caller-supplied presence
// of preparation. The existing checker owns that policy before any legacy IO.
run("python3", ["tools/check_real_video_manifest.py", manifestPath, "--validate-case-approval", caseId]);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as CorpusManifest;
const benchmarkCase = manifest.cases.find((candidate) => candidate.id === caseId);
if (!benchmarkCase) throw new Error(`unknown real-video benchmark case ${caseId}`);

const prepareOnly = process.argv.includes("--prepare-only");
if (benchmarkCase.preparation) {
  const trim = benchmarkCase.preparation.trim;
  const approvedPlan = buildVideoSamplingPlan(
    {duration: trim.end_seconds-trim.start_seconds, videoWidth:640, videoHeight:480},
    {framesPerSecond:manifest.sampling.frames_per_second,maxFrames:manifest.sampling.max_frames},
  );
  if (JSON.stringify(approvedPlan) !== JSON.stringify(benchmarkCase.preparation.plan)) {
    throw new Error("approved preparation plan differs from existing browser sampling helper");
  }
  run("python3", ["tools/check_real_video_manifest.py", manifestPath, "--prepare-case", caseId, "--asset-root", assetRoot, "--output-root", outputRoot]);
  if (!prepareOnly) {
    run("cargo", ["run", "--locked", "--release", "-p", "video-to-3d-core", "--example", "real_video_fixture", "--", resolve(outputRoot, caseId, "frames"), resolve(outputRoot, caseId)]);
  }
  console.log(`real-video-case=${caseId} preparation=offline-ffmpeg output=${resolve(outputRoot,caseId)}`);
  process.exit(0);
}
const videoPath = containedPath(assetRoot, benchmarkCase.asset.expected_file);
if (!existsSync(videoPath)) {
  throw new Error(
    `missing benchmark asset ${videoPath}; acquire it according to benchmarks/README.md`,
  );
}

const probe = JSON.parse(
  run(
    "ffprobe",
    [
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=width,height,avg_frame_rate:format=duration",
      "-of",
      "json",
      videoPath,
    ],
    true,
  ),
) as ProbeResult;
const stream = probe.streams?.[0];
const duration = Number(probe.format?.duration);
const width = Number(stream?.width);
const height = Number(stream?.height);
const frameRate = parseFrameRate(stream?.avg_frame_rate);
if (!Number.isFinite(frameRate) || frameRate <= 0) {
  throw new Error("the selected video does not expose a usable frame rate");
}
const plan = buildVideoSamplingPlan(
  { duration, videoWidth: width, videoHeight: height },
  {
    framesPerSecond: manifest.sampling.frames_per_second,
    maxFrames: manifest.sampling.max_frames,
  },
);

const lexicalCaseOutput = resolve(outputRoot, caseId);
if (existsSync(lexicalCaseOutput) && lstatSync(lexicalCaseOutput).isSymbolicLink()) {
  throw new Error("case output cannot be a symlink for replacement");
}
const caseOutput = containedPath(outputRoot, caseId);
const framesDirectory = resolve(caseOutput, "frames");
rmSync(caseOutput, { recursive: true, force: true });
mkdirSync(framesDirectory, { recursive: true });

for (const [index, time] of plan.times.entries()) {
  const output = resolve(framesDirectory, `${String(index).padStart(3, "0")}.ppm`);
  run("ffmpeg", [
    "-hide_banner",
    "-loglevel",
    "error",
    "-ss",
    time.toFixed(6),
    "-i",
    videoPath,
    "-frames:v",
    "1",
    "-vf",
    `scale=${plan.analysisWidth}:${plan.analysisHeight}:flags=bicubic,setsar=1`,
    "-pix_fmt",
    "rgb24",
    "-y",
    output,
  ]);
}

writeFileSync(
  resolve(caseOutput, "sampling.json"),
  `${JSON.stringify(
    {
      schema_version: "video-to-3d/real-video-sampling/v1",
      case_id: caseId,
      source: benchmarkCase.asset.expected_file,
      source_metadata: { duration, width, height, frame_rate: frameRate },
      plan,
    },
    null,
    2,
  )}\n`,
);

if (!prepareOnly) {
run("cargo", [
  "run",
  "--locked",
  "--release",
  "-p",
  "video-to-3d-core",
  "--example",
  "real_video_fixture",
  "--",
  framesDirectory,
  caseOutput,
]);
}

console.log(`real-video-case=${caseId} output=${caseOutput}`);
