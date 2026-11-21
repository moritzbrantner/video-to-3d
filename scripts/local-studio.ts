import { createHash } from "node:crypto";
import { link, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { rmSync } from "node:fs";
import type { ModelCandidate, ObjectModels } from "../apps/web/src/localObjects";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve, sep } from "node:path";

async function receiveFrames(form: FormData, scratch: string) {
  const metadata = form.get("frames");
  const pixels = form.getAll("pixels");
  if (typeof metadata !== "string") { throw new Error("Missing sampled frame metadata"); }
  const plan = JSON.parse(metadata);
  if (!Array.isArray(plan) || !plan.length || pixels.length !== plan.length) {
    throw new Error("Pixel files must match the sampled frames");
  }
  const frames = [];
  for (const [index, entry] of pixels.entries()) {
    const { width, height, time } = plan[index];
    if (!(entry instanceof Blob) || !Number.isSafeInteger(width) || width < 1 ||
        !Number.isSafeInteger(height) || height < 1 || !Number.isFinite(time) || time < 0 ||
        entry.size !== width * height * 4) {
      throw new Error("Invalid sampled frame");
    }
    const bytes = Buffer.from(await entry.arrayBuffer());
    const file = index + ".rgba";
    await writeFile(join(scratch, file), bytes);
    frames.push({ frame_index: index, pixels: file, width, height, timestamp_seconds: time,
      content_hash: "sha256:" + createHash("sha256").update(bytes).digest("hex") });
  }
  const manifest = join(scratch, "keyframes.json");
  await writeFile(manifest, JSON.stringify({ schema_version: 1, frames }));
  return { manifest, frames };
}

async function registerCameras(request: Request, scratch: string, registrations: Map<string, string>) {
  const { manifest } = await receiveFrames(await request.formData(), scratch);
  const output = join(scratch, "registered-geometry.json");
  const sidecar = process.env.VIDEO_TO_3D_CAMERA_SIDECAR ?? resolve("tools/colmap-sparse/run.sh");
  const log = Bun.file(join(scratch, "camera-registration.log"));
  const dense = process.env.VIDEO_TO_3D_COLMAP_DENSE !== "0";
  const args = dense ? [manifest, output, join(scratch, "colmap")] : [manifest, output];
  const child = Bun.spawn([sidecar, ...args], { stdout: "ignore", stderr: log });
  if (await child.exited !== 0) {
    const diagnostic = await log.slice(-4096).text();
    throw new Error(diagnostic.trim().split("\n").slice(-2).join(" ") || "Camera registration failed");
  }
  const headers = new Headers({ "Content-Type": "application/json" });
  if (dense && await Bun.file(join(scratch, "colmap/selected-model.txt")).exists()) {
    const token = crypto.randomUUID();
    registrations.set(token, scratch);
    setTimeout(() => {
      if (registrations.delete(token)) {
        void rm(scratch, { recursive: true, force: true }).catch(console.error);
      }
    }, 20 * 60_000).unref();
    headers.set("X-Video-To-3D-Registration", token);
  }
  return new Response(await Bun.file(output).arrayBuffer(), { headers });
}

async function detect(request: Request, scratch: string, savedInventory?: string): Promise<Response> {
  const labels = (process.env.VIDEO_TO_3D_OBJECT_LABELS ?? "").split(",").map(label => label.trim()).filter(Boolean);
  const { manifest, frames } = await receiveFrames(await request.formData(), scratch);
  if (savedInventory && await Bun.file(savedInventory).exists()) {
    const bytes = await Bun.file(savedInventory).arrayBuffer();
    const saved = JSON.parse(new TextDecoder().decode(bytes));
    const matches = saved.frames?.length === frames.length && frames.every((frame, index) => {
      const stored = saved.frames[index];
      return stored?.frame_index === index && stored.content_hash === frame.content_hash &&
        stored.width === frame.width && stored.height === frame.height;
    });
    if (!matches || (labels.length > 0 && JSON.stringify(saved.vocabulary?.terms) !== JSON.stringify(labels)) ||
        (labels.length === 0 && saved.vocabulary?.source === "project")) {
      throw new Error("Existing detection file belongs to different frames or object labels. Choose a fresh output directory.");
    }
    // The browser also validates this document through the same Rust contract as manual imports.
    return new Response(bytes, { headers: { "Content-Type": "application/json" } });
  }
  const inventory = join(scratch, "objects.json");
  const child = Bun.spawn(["cargo", "run", "--quiet", "--release", "-p", "video-to-3d-cli", "--", "detect", manifest, ...labels], {
    env: { ...process.env, VIDEO_TO_3D_SEMANTIC_SIDECAR: process.env.VIDEO_TO_3D_SEMANTIC_SIDECAR
      ?? resolve("tools/semantic-sidecar/video-to-3d-semantic-sidecar") },
    stdout: Bun.file(inventory), stderr: "inherit",
  });
  if (await child.exited !== 0) {
    const result = await Bun.file(inventory).text();
    throw new Error(result ? JSON.parse(result).error : "Unable to start native object detection");
  }
  // Publish a complete Rust-validated artifact before the browser receives it. A
  // disconnected browser can resume with the exact same frame evidence.
  if (savedInventory) { await link(inventory, savedInventory); }
  return new Response(await Bun.file(inventory).arrayBuffer(), { headers: { "Content-Type": "application/json" } });
}

type ModelRun = {
  directory: string;
  models: ObjectModels;
  assets: Map<string, string>;
};

async function nativeObjects(args: string[], scratch: string) {
  const report = join(scratch, "native-result.json");
  const child = Bun.spawn(["cargo", "run", "--quiet", "--release", "-p", "video-to-3d-cli", "--", ...args], {
    env: { ...process.env, VIDEO_TO_3D_OBJECT_GENERATION_SIDECAR: process.env.VIDEO_TO_3D_OBJECT_GENERATION_SIDECAR
      ?? resolve("tools/semantic-sidecar/video-to-3d-object-generation-sidecar") },
    stdout: Bun.file(report), stderr: "inherit",
  });
  if (await child.exited !== 0) {
    const message = await Bun.file(report).text();
    throw new Error(message ? JSON.parse(message).error : "The native object reconstruction command failed");
  }
}

async function loadModels(directory: string, publishedDirectory = directory): Promise<ModelRun> {
  const [generation, placement, references, multiview] = await Promise.all(["generation", "placement", "references", "multiview"]
    .map(name => Bun.file(join(directory, name + ".json")).json()));
  const token = crypto.randomUUID();
  const assets = new Map<string, string>();
  const candidates: ModelCandidate[] = [];
  async function meshUrl(output?: { path: string }) {
    if (!output) { return null; }
    const file = resolve(directory, output.path);
    if (!file.startsWith(directory + sep) || !file.endsWith(".glb") || !await Bun.file(file).exists()) {
      throw new Error("Reconstructed object mesh is missing or outside the run directory");
    }
    const index = String(assets.size);
    assets.set(index, resolve(publishedDirectory, output.path));
    return `/local/models/${token}/${index}.glb`;
  }
  const primary = new Map<string, { frame_index: number; crop: [number, number, number, number] }>(
    references.packages.map((pkg: {object: string; primary: {frame_index: number; crop: [number, number, number, number]}}) => [pkg.object, pkg.primary]));
  for (const item of multiview.objects) {
    const view = primary.get(item.object);
    candidates.push({ method: "multiview", object: item.object, label: item.label,
      frame: view?.frame_index ?? item.views[0] ?? null, crop: view?.crop ?? null, triangles: item.output?.triangles ?? 0, meshUrl: await meshUrl(item.output),
      placement: item.validation?.verdict ?? "unplaced", reason: item.reason || item.validation?.reason || "" });
  }
  for (const item of generation.objects) {
    const { object, label, frame_index } = item.request;
    const decision = placement.placements.find((entry: { object: string }) => entry.object === object)?.validation;
    const unplaced = placement.unplaced.find((entry: { object: string }) => entry.object === object);
    candidates.push({ method: "triposr", object, label, frame: frame_index, crop: item.request.crop, triangles: item.output?.triangles ?? 0,
      meshUrl: await meshUrl(item.output), placement: decision?.verdict ?? "unplaced", reason: item.failure ?? decision?.reason ?? unplaced?.reason ?? "" });
  }
  return { directory: publishedDirectory, assets, models: { token, candidates, unpackaged: references.unpackaged.length,
    unavailableReason: generation.unavailable_reason ?? references.unavailable_reason } };
}

async function generate(request: Request, scratch: string, modelRoot: string, persistent: boolean) {
  const form = await request.formData();
  const { manifest, frames } = await receiveFrames(form, scratch);
  const inventory = form.get("inventory");
  const geometry = form.get("geometry");
  if (!(inventory instanceof Blob) || !(geometry instanceof Blob)) {
    throw new Error("Object reconstruction requires a detection inventory and calibrated geometry");
  }
  const [inventoryText, geometryText] = await Promise.all([inventory.text(), geometry.text()]);
  const identity = createHash("sha256").update("studio-object-models/4\0").update(inventoryText)
    .update("\0").update(geometryText).update("\0").update(frames.map(f => f.content_hash).join("\n")).digest("hex");
  const directory = persistent ? modelRoot : join(modelRoot, identity);
  const identityFile = Bun.file(join(directory, "inputs.sha256"));
  if (await identityFile.exists()) {
    if ((await identityFile.text()).trim() !== identity) {
      throw new Error("Existing object models belong to another reconstruction. Choose a fresh output directory.");
    }
    return loadModels(directory);
  }
  const staging = join(scratch, "models");
  await mkdir(staging);
  const inventoryFile = join(scratch, "objects.json");
  const geometryFile = join(staging, "geometry.json");
  await Promise.all([writeFile(inventoryFile, inventoryText), writeFile(geometryFile, geometryText)]);
  await nativeObjects(["generate-objects", manifest, inventoryFile, geometryFile, staging], scratch);
  const run = await loadModels(staging, directory);
  await writeFile(join(staging, "inputs.sha256"), identity + "\n");
  try {
    await rename(staging, directory);
  } catch (error) {
    // A second converter can finish the same clip while this one is generating.
    if (await identityFile.exists() && (await identityFile.text()).trim() === identity) {
      return loadModels(directory);
    }
    throw error;
  }
  return run;
}

export async function startLocalStudio(savedInventory?: string) {
  const site = resolve("apps/web/out");
  const index = Bun.file(join(site, "index.html"));
  if (!await index.exists()) { throw new Error("Build the browser app with bun run build first"); }
  const studioPage = (html: string) => new Response(
    html.replace("</head>", '<meta name="video-to-3d-local-studio" content="1"></head>'),
    { headers: { "Content-Type": "text/html" } },
  );
  const html = await index.text();
  const temporary = await mkdtemp(join(savedInventory ? dirname(savedInventory) : tmpdir(), ".video-to-3d-studio-"));
  const modelRoot = savedInventory ? join(dirname(savedInventory), basename(savedInventory, ".objects.json") + ".models") : temporary;
  const modelRuns = new Map<string, ModelRun>();
  const registrations = new Map<string, string>();
  const cleanup = () => rmSync(temporary, { recursive: true, force: true });
  process.once("exit", cleanup);
  let pending: Promise<void> = Promise.resolve();
  const server = Bun.serve({
    hostname: "127.0.0.1", port: 0, maxRequestBodySize: 160 * 1024 * 1024,
    async fetch(request, server) {
      const url = new URL(request.url);
      if (url.host !== server.url.host) { return new Response(null, { status: 403 }); }
      const library = /^\/local\/models\/([\w-]+)\/library\.glb$/.exec(url.pathname);
      if (library && (request.method === "GET" || request.method === "HEAD")) {
        const run = modelRuns.get(library[1]);
        if (!run || run.assets.size === 0) { return new Response(null, { status: 404 }); }
        server.timeout(request, 0);
        const destination = savedInventory
          ? join(dirname(savedInventory), basename(savedInventory, ".objects.json") + ".object-library.glb")
          : join(run.directory, "object-library.glb");
        try {
          const build = pending.then(async () => {
            if (await Bun.file(destination).exists()) { return; }
            const scratch = await mkdtemp(join(temporary, "library-"));
            try {
              const libraryFile = join(scratch, "library.glb");
              await nativeObjects(["object-library", run.directory, libraryFile], scratch);
              await link(libraryFile, destination);
            } finally {
              await rm(scratch, { recursive: true, force: true });
            }
          });
          pending = build.then(() => {}, () => {});
          await build;
          return new Response(request.method === "HEAD" ? null : Bun.file(destination), {
            headers: { "Content-Type": "model/gltf-binary", "Content-Disposition": 'attachment; filename="object-library.glb"' },
          });
        } catch (error) {
          return new Response(error instanceof Error ? error.message : String(error), { status: 422 });
        }
      }
      const mesh = /^\/local\/models\/([\w-]+)\/(\d+)\.glb$/.exec(url.pathname);
      if (mesh && request.method === "GET") {
        const path = modelRuns.get(mesh[1])?.assets.get(mesh[2]);
        return path ? new Response(Bun.file(path), { headers: { "Content-Type": "model/gltf-binary" } })
          : new Response(null, { status: 404 });
      }
      const dense = /^\/local\/dense\/([a-f0-9-]+)$/.exec(url.pathname);
      if (dense && request.method === "POST" && request.headers.get("origin") === server.url.origin) {
        const scratch = registrations.get(dense[1]);
        if (!scratch) { return new Response("Registration expired", { status: 404 }); }
        registrations.delete(dense[1]);
        server.timeout(request, 0);
        const run = pending.then(async () => {
          const output = join(scratch, "observed.glb");
          const log = Bun.file(join(scratch, "dense.log"));
          const command = resolve("tools/colmap-dense/run.sh");
          const args = [join(scratch, "keyframes.json"), join(scratch, "registered-geometry.json"),
            join(scratch, "colmap"), output];
          const inventory = await request.text();
          if (inventory) {
            const inventoryPath = join(scratch, "objects.json");
            await writeFile(inventoryPath, inventory);
            args.push(inventoryPath);
          }
          const child = Bun.spawn([command, ...args], { stdout: "pipe", stderr: log });
          const [code, document] = await Promise.all([child.exited, new Response(child.stdout).text()]);
          if (code !== 0) {
            const diagnostic = (await log.slice(-4096).text()).trim().split("\n").at(-1);
            throw new Error(diagnostic || "COLMAP dense reconstruction failed");
          }
          const report = JSON.parse(document);
          const headers = new Headers({ "Content-Type": "model/gltf-binary" });
          if (Number.isSafeInteger(report.triangles) && report.triangles > 0) {
            headers.set("X-Video-To-3D-Dense-Triangles", String(report.triangles));
          }
          if (Number.isSafeInteger(report.textured_triangles) && Number.isSafeInteger(report.object_surfaces)) {
            headers.set("X-Video-To-3D-Dense-Report",
              `${report.textured_triangles} textured triangles and ${report.object_surfaces} observed object surfaces.`);
          }
          return new Response(await Bun.file(output).arrayBuffer(), { headers });
        });
        pending = run.then(() => {}, () => {});
        try { return await run; }
        catch (error) { return new Response(String(error), { status: 422 }); }
        finally { await rm(scratch, { recursive: true, force: true }); }
      }
      const completion = /^\/local\/complete\/([\w-]+)$/.exec(url.pathname);
      if (url.pathname === "/local/register-cameras" || url.pathname === "/local/objects" || url.pathname === "/local/models" || completion) {
        if (request.method !== "POST" || request.headers.get("origin") !== server.url.origin) {
          return new Response(null, { status: 403 });
        }
        server.timeout(request, 0);
        const scratch = await mkdtemp(join(temporary, "frames-"));
        try {
          // A single studio shares one GPU between uploads and browser tabs.
          const result = pending.then(async () => {
            if (url.pathname === "/local/register-cameras") { return registerCameras(request, scratch, registrations); }
            if (completion) {
              const run = modelRuns.get(completion[1]);
              if (!run) { throw new Error("The object reconstruction session has expired"); }
              const base = join(scratch, "observed.glb");
              const output = join(scratch, "completed.glb");
              await Bun.write(base, await request.arrayBuffer());
              await nativeObjects(["complete-scene", base, run.directory, output], scratch);
              return new Response(await Bun.file(output).arrayBuffer(), { headers: { "Content-Type": "model/gltf-binary" } });
            }
            if (url.pathname === "/local/models") {
              const run = await generate(request, scratch, modelRoot, savedInventory !== undefined);
              modelRuns.set(run.models.token, run);
              return Response.json(run.models);
            }
            return detect(request, scratch, savedInventory);
          });
          pending = result.then(() => {}, () => {});
          return await result;
        } catch (error) {
          return new Response(error instanceof Error ? error.message : String(error), { status: 422 });
        } finally {
          if (![...registrations.values()].includes(scratch)) {
            await rm(scratch, { recursive: true, force: true });
          }
        }
      }
      if (url.pathname === "/" || url.pathname === "/index.html") {
        return studioPage(html);
      }
      const pathname = decodeURIComponent(url.pathname);
      const asset = resolve(site, "." + pathname, pathname.endsWith("/") ? "index.html" : "");
      const file = Bun.file(asset);
      if (asset.startsWith(site + sep) && await file.exists()) {
        if (asset.endsWith(".html")) {
          return studioPage(await file.text());
        }
        return new Response(file);
      }
      return new Response(null, { status: 404 });
    },
  });
  return { url: server.url, async stop() {
    await server.stop();
    await pending;
    process.removeListener("exit", cleanup);
    cleanup();
  } };
}

if (import.meta.main) {
  const server = await startLocalStudio();
  console.log(`Local video-to-3D Studio: ${server.url.href}`);
}
