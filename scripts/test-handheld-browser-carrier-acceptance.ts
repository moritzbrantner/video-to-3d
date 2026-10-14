/** Actual browser-carrier witness; offline PPMs are not a browser pixel oracle. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, sep } from "node:path";

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  const value = process.argv[index + 1];
  assert(index >= 0 && value, `required ${name}`);
  return value;
}
const prepared = resolve(argument("--prepared-root"));
// Reuse the already installed actual Chromium automation; no application decoder dependency.
const automationModule = argument("--automation-module");
const { chromium } = await import(automationModule);
const expectationsPath = process.argv.includes("--expectations")
  ? argument("--expectations")
  : new URL("../tools/tests/fixtures/approved-handheld-expectations.json", import.meta.url);
const expectations = JSON.parse(readFileSync(expectationsPath, "utf8"));
const bundle = await Bun.build({ entrypoints: [resolve("apps/web/src/video.ts")], target: "browser" });
assert(bundle.success, "existing browser sampling module failed to bundle");
const videoModule = await bundle.outputs[0]!.text();
const carriers = new Map<string, string>();
for (const item of expectations.cases) {
  const base = resolve(prepared, item.id);
  const receipt = JSON.parse(readFileSync(resolve(base, "sampling.json"), "utf8"));
  assert.equal(receipt.case_id, item.id);
  const path = resolve(base, receipt.derived.path);
  assert(path.startsWith(base + sep), "derived carrier escapes case root");
  assert.equal(createHash("sha256").update(readFileSync(path)).digest("hex"), receipt.derived.sha256);
  assert.equal(receipt.derived.source_sha256, item.asset.sha256);
  assert.deepEqual(receipt.derived.trim, item.preparation.trim);
  carriers.set(`/${item.id}`, path);
}
const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
  const path = new URL(request.url).pathname;
  if (path === "/") {
    return new Response("<!doctype html><title>Actual handheld browser carrier acceptance</title>", { headers: { "content-type": "text/html" } });
  }
  if (path === "/video.js") {
    return new Response(videoModule, { headers: { "content-type": "text/javascript" } });
  }
  const carrier = carriers.get(path);
  if (carrier) {
    return new Response(Bun.file(carrier));
  }
  return new Response("not found", { status: 404 });
} });
const evidence: Array<Record<string, unknown>> = [];
let failed = false;
const browser = await chromium.launch({ channel: "chromium", headless: true });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.port}`);
  for (const item of expectations.cases) {
    try {
      const observed = await page.evaluate(async (id: string) => {
        const { sampleVideoWithInfo } = await import("/video.js");
        const response = await fetch(`/${id}`);
        if (!response.ok) {
          throw new Error(`actual carrier response ${response.status}`);
        }
        const bytes = await response.blob();
        const result = await sampleVideoWithInfo(new File([bytes], id));
        return { info: result.info, frames: result.frames.map((frame: { width: number; height: number; rgba: Uint8Array; time: number; presentedTime: number; presentedTimeDecoded: boolean }) => ({
          width: frame.width, height: frame.height,
          nonempty: frame.rgba.some((value) => value !== 0),
          time: frame.time, presentedTime: frame.presentedTime,
          presentedTimeDecoded: frame.presentedTimeDecoded,
        })) };
      }, item.id);
      const wantedDuration = item.preparation.trim.end_seconds - item.preparation.trim.start_seconds;
      assert(Math.abs(observed.info.duration - wantedDuration) <= 1 / 30, "browser carrier duration differs from approved interval");
      assert.equal(observed.info.displayWidth, 640);
      assert.equal(observed.info.displayHeight, 480);
      assert.equal(observed.frames.length, 18);
      for (const frame of observed.frames) {
        assert.equal(frame.width, 360);
        assert.equal(frame.height, 270);
        assert(frame.nonempty, "actual browser canvas frame has no decoded pixels");
        assert(Number.isFinite(frame.presentedTime) && frame.presentedTime >= 0 && frame.presentedTime < observed.info.duration);
      }
      evidence.push({ case_id: item.id, status: "passed", observed, browser_offline_pixel_equivalence: false });
    } catch (error) {
      failed = true;
      evidence.push({ case_id: item.id, status: "failed", error: String(error), browser_offline_pixel_equivalence: false });
    }
  }
} finally {
  await browser.close();
  server.stop();
}
const serialized = JSON.stringify({ kind: "actual-browser-carrier", browser_version: browser.version(), evidence }, null, 2);
console.log(serialized);
const reportIndex = process.argv.indexOf("--report");
if (reportIndex >= 0) {
  writeFileSync(argument("--report"), serialized + "\n");
}
assert(!failed, "approved derived carriers must decode through actual sampleVideoWithInfo");
