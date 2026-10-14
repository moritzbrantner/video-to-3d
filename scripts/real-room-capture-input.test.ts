import { expect, test } from "bun:test";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { digest, verifyCaptureInput } from "./real-room-capture-input";
import identity from "../tools/tests/fixtures/approved-real-room-capture-identity.json";

const original = process.env.VIDEO_TO_3D_REAL_ROOM_SOURCE;
const prepared = process.env.VIDEO_TO_3D_REAL_ROOM_PREPARED;
const realInput = test.skipIf(!original || !prepared);

realInput("canonical approved preparation agrees with independent carrier and receipt pins", () => {
  expect(verifyCaptureInput(original!, prepared!).expected.id).toBe("approved-controlled-room-tum");
});

for (const rewriteReceipt of [false, true]) {
  realInput("substituted carrier is rejected before browser work, receipt rewritten=" + rewriteReceipt, () => {
    const root = mkdtempSync(join(tmpdir(), "real-room-input-mutation-"));
    try {
      const base = join(root, "approved-controlled-room-tum");
      mkdirSync(base);
      const receiptBytes = readFileSync(join(prepared!, "approved-controlled-room-tum/sampling.json"));
      const receipt = JSON.parse(receiptBytes.toString("utf8"));
      // Keep the real carrier framing and exact size: length checks alone cannot
      // reject this changed copy, even if its colocated receipt is rewritten.
      const substitute = readFileSync(join(prepared!, "approved-controlled-room-tum/derived.webm"));
      substitute[substitute.length - 1]! ^= 1;
      writeFileSync(join(base, "derived.webm"), substitute);
      const changedHash = digest(substitute).slice("sha256:".length);
      const changedReceipt = rewriteReceipt
        ? Buffer.from(receiptBytes.toString("utf8").replace(receipt.derived.sha256, changedHash))
        : receiptBytes;
      expect(changedReceipt.length).toBe(receiptBytes.length);
      if (rewriteReceipt) {
        const selfConsistent = JSON.parse(changedReceipt.toString("utf8"));
        expect(selfConsistent.derived.sha256).toBe(changedHash);
        expect(selfConsistent.derived.byte_length).toBe(substitute.length);
      }
      writeFileSync(join(base, "sampling.json"), changedReceipt);
      const output = join(root, "must-not-be-captured");
      const result = Bun.spawnSync([process.execPath,
        "scripts/capture-real-room-native-acceptance.ts", "--original", original!,
        "--prepared-root", root, "--output-root", output,
        "--automation-module", join(root, "browser-must-not-be-imported.mjs")],
        { cwd: fileURLToPath(new URL("..", import.meta.url)) });
      expect(result.exitCode).not.toBe(0);
      const diagnostic = result.stderr.toString();
      expect(diagnostic).toContain(rewriteReceipt ? "sampling receipt identity changed" : "prepared carrier identity changed");
      expect(diagnostic).not.toContain("Cannot find module");
      expect(result.stdout.toString()).not.toContain("witness_hash");
      expect(existsSync(output)).toBe(false);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
}

realInput("carrier HTTP bytes remain pinned after the verified path is replaced", () => {
  const root = mkdtempSync(join(tmpdir(), "real-room-carrier-race-"));
  try {
    const base = join(root, "approved-controlled-room-tum");
    mkdirSync(base);
    const carrier = join(base, "derived.webm");
    writeFileSync(carrier, readFileSync(join(prepared!, "approved-controlled-room-tum/derived.webm")));
    writeFileSync(join(base, "sampling.json"), readFileSync(join(prepared!, "approved-controlled-room-tum/sampling.json")));
    const report = join(root, "served-bytes.json");
    const automation = join(root, "network-only-probe.mjs");
    // This is an actual HTTP byte-source test, deliberately stopped before any
    // browser evaluation. The separate real Chromium tier owns pixel evidence.
    writeFileSync(automation, `
      import {readFileSync,writeFileSync} from "node:fs";
      import {createHash} from "node:crypto";
      const hash = bytes => "sha256:" + createHash("sha256").update(bytes).digest("hex");
      export const chromium = { launch: async () => ({
        newPage: async () => ({ goto: async url => {
          const bytes = readFileSync(${JSON.stringify(carrier)});
          bytes[0] ^= 1;
          writeFileSync(${JSON.stringify(carrier)}, bytes);
          const response = await fetch(new URL("/carrier", url));
          const served = new Uint8Array(await response.arrayBuffer());
          writeFileSync(${JSON.stringify(report)}, JSON.stringify({status:response.status,onDiskHash:hash(bytes),servedHash:hash(served)}));
          throw new Error("network-only carrier probe complete");
        }}),
        close: async () => {},
      })};
    `);
    const output = join(root, "must-not-be-captured");
    const result = Bun.spawnSync([process.execPath,
      "scripts/capture-real-room-native-acceptance.ts", "--original", original!,
      "--prepared-root", root, "--output-root", output,
      "--automation-module", automation],
      { cwd: fileURLToPath(new URL("..", import.meta.url)) });
    expect(result.exitCode).not.toBe(0);
    expect(result.stderr.toString()).toContain("network-only carrier probe complete");
    const actual: unknown = JSON.parse(readFileSync(report, "utf8"));
    expect(actual).toMatchObject({
      status: 200,
      servedHash: `sha256:${identity.carrier.sha256}`,
      onDiskHash: expect.not.stringContaining(identity.carrier.sha256),
    });
    expect(existsSync(output)).toBe(false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
