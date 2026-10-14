/** Independent #179 input author: actual browser pixels, never offline PPM substitutes. */
import assert from "node:assert/strict";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { digest, verifyCaptureInput } from "./real-room-capture-input";

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  const value = process.argv[index + 1];
  assert(index >= 0 && value, `required ${name}`);
  return value;
}
const { expected, original, receiptBytes, carrierPath, carrier } = verifyCaptureInput(
  argument("--original"), argument("--prepared-root"),
);
const output = resolve(argument("--output-root"));
assert(!existsSync(output), "capture destination must be fresh");
const bundle = await Bun.build({ entrypoints: [resolve("apps/web/src/video.ts")], target: "browser" });
assert(bundle.success);
const module = await bundle.outputs[0]!.text();
const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
  switch (new URL(request.url).pathname) {
    case "/": return new Response("<!doctype html><title>Independent real-room input capture</title>", { headers: { "content-type": "text/html" } });
    case "/video.js": return new Response(module, { headers: { "content-type": "text/javascript" } });
    case "/carrier": return new Response(Bun.file(carrierPath));
    default: return new Response("not found", { status: 404 });
  }
} });
const { chromium } = await import(argument("--automation-module"));
const browser = await chromium.launch({ channel: "chromium", headless: true });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.port}`);
  const captured = await page.evaluate(async () => {
    const { sampleVideoWithInfo } = await import("/video.js");
    const response = await fetch("/carrier");
    if (!response.ok) throw new Error(`carrier HTTP ${response.status}`);
    const result = await sampleVideoWithInfo(new File([await response.blob()], "approved-room.webm"));
    return { info: result.info, frames: await Promise.all(result.frames.map(async (frame: { width: number; height: number; rgba: Uint8Array; time: number; presentedTime: number; presentedTimeDecoded: boolean }) => ({
      width: frame.width, height: frame.height, requested_time: frame.time,
      presented_time: frame.presentedTime, presented_time_decoded: frame.presentedTimeDecoded,
      browser_hash: `sha256:${Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", frame.rgba))).map((byte) => byte.toString(16).padStart(2, "0")).join("")}`,
      rgba: Array.from(frame.rgba),
    }))) };
  });
  assert.equal(captured.info.displayWidth, 640);
  assert.equal(captured.info.displayHeight, 480);
  assert(Math.abs(captured.info.duration - 86.166667) <= 1 / 30);
  assert.equal(captured.frames.length, 18);
  const frames = captured.frames.map((frame: { width: number; height: number; rgba: number[]; browser_hash: string; requested_time: number; presented_time: number; presented_time_decoded: boolean }, index: number) => {
    assert.equal(frame.width, 360);
    assert.equal(frame.height, 270);
    assert.equal(frame.rgba.length, 360 * 270 * 4);
    assert(frame.rgba.some((value) => value !== 0));
    assert.equal(digest(Uint8Array.from(frame.rgba)), frame.browser_hash, "retained bytes must equal actual canvas bytes");
    assert(Math.abs(frame.requested_time - expected.preparation.plan.times[index]) <= 1 / 30);
    assert.equal(frame.presented_time_decoded, true, "this fixed carrier must expose decoded timestamps");
    assert(Number.isFinite(frame.presented_time) && frame.presented_time >= 0 && frame.presented_time < captured.info.duration);
    const { rgba, browser_hash, ...metadata } = frame;
    return { ...metadata, frame_index: index, path: `frames/${String(index).padStart(3, "0")}.rgba`, byte_length: rgba.length, content_hash: browser_hash };
  });
  mkdirSync(resolve(output, "frames"), { recursive: true });
  for (const [index, frame] of frames.entries()) {
    writeFileSync(resolve(output, frame.path), Uint8Array.from(captured.frames[index].rgba));
  }
  const document = JSON.stringify({
    schema_version: 1, case_id: expected.id,
    original: { content_hash: digest(original), byte_length: original.length },
    carrier: { content_hash: digest(carrier), byte_length: carrier.length },
    sampling_receipt_hash: digest(receiptBytes), approved_plan: expected.preparation.plan,
    browser_version: browser.version(), info: captured.info, frames,
    browser_offline_pixel_equivalence: false,
  }, null, 2) + "\n";
  writeFileSync(resolve(output, "browser-frames.json"), document);
  console.log(JSON.stringify({ kind: "independent-real-room-browser-capture", output, witness_hash: digest(Buffer.from(document)), browser_version: browser.version(), frames: frames.length }));
} finally {
  await browser.close();
  server.stop();
}
