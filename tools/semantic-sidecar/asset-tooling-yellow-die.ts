#!/usr/bin/env bun
// First-object acceptance adapter for the existing native generation protocol.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { pathToFileURL } from "node:url";

const protocol = "video-to-3d-object-generation-sidecar/1";
const modelId = "stabilityai/TripoSR";
const modelRevision = "5b521936b01fbe1890f6f9baed0254ab6351c04a";
function required(name: string) {
  const value = process.env[name];
  if (!value || !path.isAbsolute(value)) {
    throw new Error(`${name} must name an absolute local path`);
  }
  return value;
}

type Candidate = {
  request_id: string;
  label: string;
  object: string;
  observed_points: number;
  frame_index: number;
  timestamp_seconds: number;
  source_frame_hash: string;
  rgba_path: string;
  mask_hash: string;
  output_path: string;
};

const [requestPath, responsePath] = process.argv.slice(2);
if (!requestPath || !responsePath) {
  throw new Error("expected native sidecar request and response paths");
}
const request = JSON.parse(await readFile(requestPath, "utf8"));
if (request.protocol !== protocol) {
  throw new Error("unsupported native generation protocol");
}
const candidates: Candidate[] = request.requests
  .filter((entry: Candidate) => /\b(die|dice|cube)\b/i.test(entry.label))
  .sort((left: Candidate, right: Candidate) => right.observed_points - left.observed_points);
const [candidate, runnerUp] = candidates;
if (!candidate || runnerUp?.observed_points === candidate.observed_points) {
  throw new Error("no unique best-supported yellow-die package");
}

const assetTooling = required("VIDEO_TO_3D_ASSET_TOOLING");
const modelPath = required("VIDEO_TO_3D_TRIPOSR_BUNDLE");
const expectedHash = process.env.VIDEO_TO_3D_TRIPOSR_BUNDLE_SHA256;
if (!expectedHash || !/^[a-f0-9]{64}$/.test(expectedHash)) {
  throw new Error("VIDEO_TO_3D_TRIPOSR_BUNDLE_SHA256 must be the pinned lowercase bundle hash");
}
const evidenceRoot = required("VIDEO_TO_3D_OBJECT_EVIDENCE");
await mkdir(evidenceRoot, { recursive: true });
const sourceHash = createHash("sha256").update(await readFile(candidate.rgba_path)).digest("hex");
if (`sha256:${sourceHash}` !== candidate.source_frame_hash) {
  throw new Error("reference-frame RGBA SHA-256 mismatch");
}
const viewPath = path.join(evidenceRoot, "view.png");
const prepared = Bun.spawnSync(
  [
    "python3", "-c",
    `import json, sys, torch
from semantic_sidecar.object_generation import object_image
if not torch.cuda.is_available():
    raise SystemExit("TripoSR acceptance requires CUDA")
entry = next(item for item in json.load(open(sys.argv[1]))["requests"] if item["request_id"] == sys.argv[2])
object_image(entry).save(sys.argv[3])
print(json.dumps({"name": "pytorch", "version": torch.__version__, "device": "cuda", "device_name": torch.cuda.get_device_name(0)}))`,
    requestPath, candidate.request_id, viewPath,
  ],
  { cwd: import.meta.dir, timeout: 60_000, stderr: "inherit" },
);
if (prepared.exitCode !== 0) {
  throw new Error("masked crop preparation or CUDA probe failed");
}
const runtime = JSON.parse(prepared.stdout.toString());
const { storeAssetObject, resolveAssetObject } = await import(
  pathToFileURL(path.join(assetTooling, "src/asset-store.ts")).href
);
const { createImageTo3DOperationExecutor } = await import(
  pathToFileURL(path.join(assetTooling, "src/image-to-3d-operation.ts")).href
);
const modelBytes = await readFile(modelPath);
if (createHash("sha256").update(modelBytes).digest("hex") !== expectedHash) {
  throw new Error("TripoSR bundle SHA-256 mismatch");
}
const model = (await storeAssetObject(evidenceRoot, {
  bytes: modelBytes, kind: "model", mediaType: "application/zip",
})).asset;
const view = (await storeAssetObject(evidenceRoot, {
  bytes: await readFile(viewPath), kind: "image", mediaType: "image/png",
})).asset;
// Sample whole-device used memory, including the driver and other processes.
function deviceMemoryMiB() {
  const query = Bun.spawnSync([
    "nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits",
  ], { timeout: 2_000 });
  if (query.exitCode !== 0) {
    return undefined;
  }
  const matches = query.stdout.toString().trim().split("\n").flatMap((row) => {
    const [name, used] = row.split(",").map((cell) => cell.trim());
    const mib = Number(used);
    return name === runtime.device_name && Number.isFinite(mib) ? [mib] : [];
  });
  return matches.length === 1 ? matches[0] : undefined;
}
const baselineMemoryMiB = deviceMemoryMiB();
let peakMemoryMiB = baselineMemoryMiB;
let memorySamples = 0;
const sampler = setInterval(() => {
  const used = deviceMemoryMiB();
  if (used !== undefined) {
    peakMemoryMiB = Math.max(peakMemoryMiB ?? used, used);
    memorySamples += 1;
  }
}, 500);
const started = performance.now();
const result = await createImageTo3DOperationExecutor({ cache: false })(evidenceRoot, {
  parameters: {
    provider: {
      id: "triposr", modelId, modelRevision, assets: ["model"],
      parameters: {
        bundleId: "triposr-local-bundle", preprocessMode: "prepared",
        device: "cuda", chunkSize: 8192, mcResolution: 256, deterministicAlgorithms: false,
      },
    },
    views: [{ id: `frame-${candidate.frame_index}`, viewpoint: "unspecified" }],
    description: candidate.label, quality: "standard", requirements: { materials: "geometry" },
  },
  inputs: { views: [view], "provider-assets": [model] },
}).finally(() => clearInterval(sampler));
const elapsedSeconds = (performance.now() - started) / 1000;
const output = result.outputs.output;
const provider = result.observations.provider;
if (!Number.isSafeInteger(provider?.vertexCount) || provider.vertexCount <= 0
    || !Number.isSafeInteger(provider?.faceCount) || provider.faceCount <= 0) {
  throw new Error("TripoSR did not report a nonempty mesh");
}
const evidence = {
  object: candidate.object,
  source_frame_index: candidate.frame_index,
  source_frame_hash: candidate.source_frame_hash,
  timestamp_seconds: candidate.timestamp_seconds,
  mask_hash: candidate.mask_hash,
  runtime,
  model_license: "MIT",
  sampled_device_memory: {
    baseline_mib: baselineMemoryMiB ?? null,
    peak_mib: memorySamples > 0 ? peakMemoryMiB : null,
    samples: memorySamples,
  },
  output,
  observations: result.observations,
  elapsed_seconds: elapsedSeconds,
};
await writeFile(candidate.output_path, await resolveAssetObject(evidenceRoot, output));
await writeFile(path.join(evidenceRoot, "generation.json"), JSON.stringify(evidence, null, 2));
await writeFile(responsePath, JSON.stringify({
  protocol,
  status: "available",
  models: [{ role: "generator", id: modelId, version: modelRevision, url: `https://huggingface.co/${modelId}` }],
  runtime,
  results: [{
    request_id: candidate.request_id, status: "ok",
    vertices: provider.vertexCount, triangles: provider.faceCount, evidence,
  }, ...request.requests.filter((entry: Candidate) => entry !== candidate).map((entry: Candidate) => ({
    request_id: entry.request_id, status: "failed", reason: "outside the first yellow-die inference slice",
  }))],
}));
