/** Acceptance-only trust boundary for the independently verified #188 preparation. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import expectations from "../tools/tests/fixtures/approved-controlled-room-expectations.json";
import identity from "../tools/tests/fixtures/approved-real-room-capture-identity.json";

export const digest = (bytes: Uint8Array) =>
  `sha256:${createHash("sha256").update(bytes).digest("hex")}`;

export function verifyCaptureInput(originalPath: string, preparedRoot: string) {
  const expected = expectations.cases[0]!;
  assert.equal(identity.case_id, expected.id);
  assert.deepEqual(identity.original, {
    sha256: expected.asset.sha256,
    byte_length: expected.asset.byte_length,
  });
  const original = readFileSync(originalPath);
  assert.equal(digest(original), `sha256:${identity.original.sha256}`);
  assert.equal(original.length, identity.original.byte_length);
  const base = resolve(preparedRoot, expected.id);
  const receiptBytes = readFileSync(resolve(base, "sampling.json"));
  // A colocated receipt is untrusted until this external, repository-owned pin agrees.
  assert.equal(digest(receiptBytes), `sha256:${identity.sampling_receipt.sha256}`,
    "sampling receipt identity changed");
  assert.equal(receiptBytes.length, identity.sampling_receipt.byte_length);
  const receipt = JSON.parse(receiptBytes.toString("utf8"));
  assert.equal(receipt.case_id, expected.id);
  assert.equal(receipt.original.sha256, expected.asset.sha256);
  assert.equal(receipt.original.byte_length, original.length);
  assert.deepEqual(receipt.plan, expected.preparation.plan);
  assert.deepEqual(receipt.derived.trim, expected.preparation.trim);
  assert.equal(receipt.derived.source_sha256, expected.asset.sha256);
  assert.equal(receipt.derived.path, identity.carrier.path);
  const carrierPath = resolve(base, identity.carrier.path);
  assert(carrierPath.startsWith(base + sep));
  const carrier = readFileSync(carrierPath);
  assert.equal(digest(carrier), `sha256:${identity.carrier.sha256}`,
    "prepared carrier identity changed");
  assert.equal(carrier.length, identity.carrier.byte_length);
  assert.equal(receipt.derived.sha256, identity.carrier.sha256);
  assert.equal(receipt.derived.byte_length, identity.carrier.byte_length);
  return { expected, original, receiptBytes, carrierPath, carrier };
}
