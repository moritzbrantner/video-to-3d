// Reuse the shipped application for browser decoding, reconstruction, and export.
import assert from "node:assert/strict";
import { constants } from "node:fs";
import { copyFile, mkdir, stat } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { startLocalStudio } from "./local-studio";
import { chromium } from "playwright-core";

const [video, outputDirectory = "target/video-to-scene", ...extra] = process.argv.slice(2);
assert(video && extra.length === 0, "Usage: bun run reconstruct <video> [output-directory]");
assert((await stat(video)).isFile(), `Expected a local video file: ${video}`);
const stem = basename(video, extname(video));
await mkdir(outputDirectory, { recursive: true });
const inventory = resolve(outputDirectory, stem + ".objects.json");
for (const extension of [".glb", ".ply"]) {
  assert(!await Bun.file(resolve(outputDirectory, stem + extension)).exists(), "Scene already exists. Choose a fresh output directory.");
}
const server = await startLocalStudio(inventory);
async function convert() {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    await page.route(/^https?:/, route => {
      return new URL(route.request().url()).origin === server.url.origin
        ? route.continue()
        : route.abort();
    });
    await page.goto(server.url.href);
    const input = page.locator('input[type="file"][accept*="video"]');
    await input.setInputFiles(resolve(video));
    await page.locator(".video-run").waitFor();
    await page.waitForFunction(() => {
      const picker = document.querySelector<HTMLInputElement>('input[type="file"][accept*="video"]');
      return picker && !picker.disabled;
    }, undefined, { timeout: 3_600_000 });

    const registration = page.getByTestId("camera-registration");
    if (await registration.count()) { console.log(await registration.innerText()); }
    const evidenceStatus = page.getByTestId("object-inventory-loaded");
    if (await evidenceStatus.count()) {
      const result = await Bun.file(inventory).json();
      console.log(result.status === "available"
        ? `Detected ${result.observations.length} observations and ${result.tracks.length} tracks: ${inventory}`
        : `Object detection unavailable: ${result.unavailable_reason}`);
    } else {
      const diagnostic = await page.getByRole("region", { name: "Detected objects" }).innerText();
      console.warn("Object detection: " + diagnostic);
      if (await Bun.file(inventory).exists()) {
        throw new Error("Saved object evidence could not be matched to this video. Choose a fresh output directory.");
      }
    }
    const surface = page.getByRole("button", { name: /^Export (textured|completed scene) GLB$/ });
    const models = page.getByTestId("object-models-ready");
    const modelCount = await models.count() ? Number(await models.getAttribute("data-generated")) : 0;
    if (await models.count()) {
      console.log(`${await models.innerText()}: ${resolve(outputDirectory, stem + ".models")}`);
    } else {
      const failure = page.getByRole("region", { name: "Reconstructed objects" }).getByRole("alert");
      if (await failure.count()) {
        console.warn("Object reconstruction failed: " + await failure.innerText());
        // Keep the observed export, but signal an incomplete reconstruction to automation.
        process.exitCode = 1;
      }
    }
    if (modelCount > 0) {
      const href = await page.getByRole("link", { name: "Download object library (unplaced grid)" }).getAttribute("href");
      assert(href, "Missing object library download");
      const response = await fetch(new URL(href, server.url), { method: "HEAD" });
      assert(response.ok, `Unable to export object library (HTTP ${response.status})`);
      console.log("Unplaced object library: " + resolve(outputDirectory, stem + ".object-library.glb"));
    }
    const points = page.getByRole("button", { name: "Export point cloud (PLY)" });
    const hasSurface = await surface.count() > 0 && await surface.isEnabled();
    const button = hasSurface ? surface : points;
    if (!await button.count() || !await button.isEnabled()) {
      if (modelCount > 0) {
        console.log("Individual object meshes are available. Scene geometry could not be exported.");
        return;
      }
      const error = await page.locator(".notice-error").allTextContents();
      throw new Error([await page.locator(".status-line").innerText(), ...error].join(" "));
    }
    async function tryExport(exportButton: typeof button) {
      const failure = page.locator(".method-note").filter({ hasText: "Material export failed:" });
      const result = Promise.race([
        page.waitForEvent("download", { timeout: 120_000 }).then(download => ({ download })),
        failure.waitFor({ timeout: 120_000 }).then(async () => ({ error: await failure.innerText() })),
      ]);
      await exportButton.click();
      return result;
    }

    let exported = await tryExport(button);
    if ("error" in exported) {
      const observed = page.getByRole("button", { name: "Export observed-only GLB", exact: true });
      if (!await observed.count() || !await observed.isEnabled()) throw new Error(exported.error);
      console.warn(exported.error + " Writing the observed scene.");
      process.exitCode = 1;
      // The page clears the old error when the observed export starts.
      const [download] = await Promise.all([
        page.waitForEvent("download", { timeout: 120_000 }), observed.click(),
      ]);
      exported = { download };
    }
    const output = resolve(outputDirectory, stem + (hasSurface ? ".glb" : ".ply"));
    // Preserve existing scenes, including any edits made after importing the first export.
    const temporary = await exported.download.path();
    assert(temporary, "Browser download failed");
    await copyFile(temporary, output, constants.COPYFILE_EXCL);
    console.log(`${await page.locator(".status-line strong").innerText()}: ${output}`);
  } finally {
    await browser.close();
  }
}
try {
  try {
    await convert();
  } catch (error) {
    if (!(error instanceof Error) || !/page.*(?:closed|crashed)/i.test(error.message)) {
      throw error;
    }
    console.warn("The browser closed during conversion. Reopening the video and reusing matching local detections.");
    await convert();
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
} finally {
  await server.stop();
}
