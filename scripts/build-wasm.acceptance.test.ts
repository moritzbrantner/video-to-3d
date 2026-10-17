// Independent acceptance for #192: scripts/build-wasm.sh must bind wasm-bindgen
// to the artifacts produced in the Cargo target directory actually chosen for
// the build, for both the main and the feature WASM crates.
//
// The script runs from a disposable copy of the repository layout with
// PATH-shimmed `cargo` and `wasm-bindgen` stubs, so no Rust compilation runs.
// The cargo stub resolves its effective target directory like Cargo does
// (--target-dir > CARGO_TARGET_DIR > CARGO_BUILD_TARGET_DIR > <manifest dir>/target,
// relative values against the invocation cwd) and writes a unique per-run
// artifact there. The wasm-bindgen stub records the exact input path and bytes it
// received and, like the real tool, fails when the input is missing.
//
// This boundary test does not replace the real Rust -> WASM check described in
// the PR (#192): CARGO_TARGET_DIR=$(mktemp -d) bash scripts/build-wasm.sh, then
// the WASM/browser contract.
import { afterEach, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const TRIPLE = "wasm32-unknown-unknown";
const MAIN_LIB = "video_to_3d_wasm";
const FEATURE_LIB = "video_to_3d_feature_wasm";
const FEATURE_CRATE = "crates/video-to-3d-feature-wasm";

const CARGO_STUB = String.raw`#!/usr/bin/env bun
const { appendFileSync, mkdirSync, writeFileSync } = require("node:fs");
const { basename, dirname, join, resolve } = require("node:path");
const args = process.argv.slice(2);
const log = process.env.STUB_LOG;
const sub = args[0];
const value = (name) => {
  for (let i = 0; i < args.length; i++) {
    if (args[i] === name) return args[i + 1];
    if (args[i].startsWith(name + "=")) return args[i].slice(name.length + 1);
  }
  return undefined;
};
const manifest = resolve(value("--manifest-path") ?? join(process.cwd(), "Cargo.toml"));
const explicit = value("--target-dir");
const envTarget = process.env.CARGO_TARGET_DIR || process.env.CARGO_BUILD_TARGET_DIR;
const target = explicit !== undefined ? resolve(explicit) : envTarget ? resolve(envTarget) : join(dirname(manifest), "target");
const record = { tool: "cargo", args, cwd: process.cwd(), env_CARGO_TARGET_DIR: process.env.CARGO_TARGET_DIR ?? null, manifest, target };
if (sub === "metadata") {
  appendFileSync(log, JSON.stringify(record) + "\n");
  process.stdout.write(JSON.stringify({ packages: [], workspace_members: [], target_directory: target, workspace_root: dirname(manifest), version: 1 }) + "\n");
  process.exit(0);
}
if (sub !== "build") {
  console.error("cargo stub: unsupported subcommand " + sub + "; extend the acceptance stub (requires independent re-verification)");
  process.exit(97);
}
const pkg = value("-p") ?? value("--package") ?? basename(dirname(manifest));
const lib = pkg.replace(/-/g, "_");
const triple = value("--target") ?? "host";
const profile = args.includes("--release") ? "release" : "debug";
const artifact = join(target, triple, profile, lib + ".wasm");
const nonce = process.env.STUB_NONCE + ":" + lib + ":" + Math.random().toString(36).slice(2);
const omit = (process.env.STUB_OMIT ?? "").split(",").includes(lib);
if (!omit) {
  mkdirSync(dirname(artifact), { recursive: true });
  writeFileSync(artifact, nonce);
}
appendFileSync(log, JSON.stringify({ ...record, lib, artifact, nonce: omit ? null : nonce }) + "\n");
const format = value("--message-format");
if (format && format.startsWith("json")) {
  process.stdout.write(JSON.stringify({ reason: "compiler-artifact", package_id: pkg, target: { name: lib, kind: ["cdylib", "rlib"], crate_types: ["cdylib", "rlib"] }, filenames: [artifact, join(target, triple, profile, "lib" + lib + ".rlib")], fresh: false }) + "\n");
  process.stdout.write(JSON.stringify({ reason: "build-finished", success: true }) + "\n");
}
`;

const BINDGEN_STUB = String.raw`#!/usr/bin/env bun
const { appendFileSync, existsSync, readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const args = process.argv.slice(2);
const flags = new Set(["--target", "--out-dir", "--out-name"]);
let input;
for (let i = 0; i < args.length; i++) {
  if (flags.has(args[i])) { i++; continue; }
  if (!args[i].startsWith("--")) input = args[i];
}
const path = input === undefined ? null : resolve(input);
const content = path && existsSync(path) ? readFileSync(path, "utf8") : null;
appendFileSync(process.env.STUB_LOG, JSON.stringify({ tool: "wasm-bindgen", args, cwd: process.cwd(), input: path, content }) + "\n");
if (content === null) {
  console.error("wasm-bindgen stub: missing input " + input);
  process.exit(1);
}
`;

type CargoCall = {
  tool: "cargo";
  args: string[];
  cwd: string;
  manifest: string;
  target: string;
  lib?: string;
  artifact?: string;
  nonce?: string | null;
};
type BindgenCall = { tool: "wasm-bindgen"; args: string[]; cwd: string; input: string | null; content: string | null };
type Call = CargoCall | BindgenCall;

type Run = {
  status: number | null;
  output: string;
  calls: Call[];
  root: string;
  cwd: string;
};

const sandboxes: string[] = [];
afterEach(() => {
  for (const dir of sandboxes.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function sandbox() {
  const base = realpathSync(mkdtempSync(join(tmpdir(), "build-wasm-192-")));
  sandboxes.push(base);
  const root = join(base, "repo");
  mkdirSync(join(root, FEATURE_CRATE), { recursive: true });
  // Copy the script directory under test unchanged; ROOT is derived from it.
  cpSync(join(REPO, "scripts"), join(root, "scripts"), { recursive: true });
  writeFileSync(join(root, "Cargo.toml"), readFileSync(join(REPO, "Cargo.toml")));
  writeFileSync(join(root, FEATURE_CRATE, "Cargo.toml"), readFileSync(join(REPO, FEATURE_CRATE, "Cargo.toml")));
  const stubs = join(base, "stubs");
  mkdirSync(stubs);
  writeFileSync(join(stubs, "cargo"), CARGO_STUB);
  writeFileSync(join(stubs, "wasm-bindgen"), BINDGEN_STUB);
  chmodSync(join(stubs, "cargo"), 0o755);
  chmodSync(join(stubs, "wasm-bindgen"), 0o755);
  const cwd = join(base, "caller");
  mkdirSync(cwd);
  return { base, root, stubs, cwd };
}

function placeStaleRootSentinels(root: string) {
  const sentinels = [
    join(root, "target", TRIPLE, "release", `${MAIN_LIB}.wasm`),
    join(root, "target", "feature-lab", TRIPLE, "release", `${FEATURE_LIB}.wasm`),
    join(root, FEATURE_CRATE, "target", TRIPLE, "release", `${FEATURE_LIB}.wasm`),
  ];
  for (const file of sentinels) {
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, `STALE-SENTINEL:${file}`);
  }
  return sentinels;
}

function runScript(
  box: ReturnType<typeof sandbox>,
  opts: { targetDir?: string; features?: string; omit?: string[] } = {},
): Run {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) if (v !== undefined) env[k] = v;
  delete env.CARGO_TARGET_DIR;
  delete env.CARGO_BUILD_TARGET_DIR;
  delete env.WASM_FEATURES;
  env.PATH = `${box.stubs}:${dirname(process.execPath)}:${env.PATH ?? ""}`;
  env.STUB_LOG = join(box.base, "calls.jsonl");
  env.STUB_NONCE = `run-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  env.STUB_OMIT = (opts.omit ?? []).join(",");
  if (opts.targetDir !== undefined) env.CARGO_TARGET_DIR = opts.targetDir;
  if (opts.features !== undefined) env.WASM_FEATURES = opts.features;
  writeFileSync(env.STUB_LOG, "");
  const result = spawnSync("bash", [join(box.root, "scripts", "build-wasm.sh")], {
    cwd: box.cwd,
    env,
    encoding: "utf8",
    timeout: 60_000,
  });
  const calls = readFileSync(env.STUB_LOG, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Call);
  return { status: result.status, output: `${result.stdout}\n${result.stderr}`, calls, root: box.root, cwd: box.cwd };
}

const builds = (run: Run) => run.calls.filter((c): c is CargoCall => c.tool === "cargo" && c.args[0] === "build");
const bindgens = (run: Run) => run.calls.filter((c): c is BindgenCall => c.tool === "wasm-bindgen");

function within(child: string, parent: string) {
  const rel = relative(parent, child);
  return rel === "" || (!rel.startsWith("..") && !rel.startsWith(sep) && rel !== "..");
}

function argAfter(args: string[], name: string) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}

function bindgenFor(run: Run, outName: string) {
  const calls = bindgens(run).filter((c) => argAfter(c.args, "--out-name") === outName);
  expect(calls.length).toBe(1);
  return calls[0]!;
}

function buildFor(run: Run, lib: string) {
  const calls = builds(run).filter((c) => c.lib === lib);
  expect(calls.length).toBe(1);
  return calls[0]!;
}

function listFiles(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? listFiles(p) : [p];
  });
}

/** Shared contract: names, out dirs, flags and feature forwarding stay as on main. */
function expectUnchangedInterface(run: Run, features?: string) {
  const main = buildFor(run, MAIN_LIB);
  expect(main.manifest).toBe(join(run.root, "Cargo.toml"));
  expect(main.args).toContain("--release");
  expect(argAfter(main.args, "--target")).toBe(TRIPLE);
  expect(argAfter(main.args, "-p")).toBe("video-to-3d-wasm");
  if (features === undefined) expect(main.args).not.toContain("--features");
  else expect(argAfter(main.args, "--features")).toBe(features);

  const feature = buildFor(run, FEATURE_LIB);
  expect(feature.manifest).toBe(join(run.root, FEATURE_CRATE, "Cargo.toml"));
  expect(feature.args).toContain("--release");
  expect(argAfter(feature.args, "--target")).toBe(TRIPLE);
  expect(feature.args).not.toContain("--features");

  const mainBind = bindgenFor(run, MAIN_LIB);
  expect(argAfter(mainBind.args, "--target")).toBe("web");
  expect(resolve(run.cwd, argAfter(mainBind.args, "--out-dir")!)).toBe(join(run.root, "apps/web/public/wasm"));
  const featureBind = bindgenFor(run, FEATURE_LIB);
  expect(argAfter(featureBind.args, "--target")).toBe("web");
  expect(resolve(run.cwd, argAfter(featureBind.args, "--out-dir")!)).toBe(join(run.root, "apps/web/public/feature-wasm"));
  expect(bindgens(run).length).toBe(2);

  // Each bindgen runs after the build that produces its input.
  expect(run.calls.indexOf(mainBind)).toBeGreaterThan(run.calls.indexOf(main));
  expect(run.calls.indexOf(featureBind)).toBeGreaterThan(run.calls.indexOf(feature));
}

/** Exact identity: bindgen consumed the bytes this run's build wrote, at the chosen target. */
function expectBoundToTarget(run: Run, chosen: string) {
  const main = buildFor(run, MAIN_LIB);
  expect(main.target).toBe(chosen);
  const mainArtifact = join(chosen, TRIPLE, "release", `${MAIN_LIB}.wasm`);
  expect(main.artifact).toBe(mainArtifact);
  const mainBind = bindgenFor(run, MAIN_LIB);
  expect(mainBind.input).toBe(mainArtifact);
  expect(mainBind.content).toBe(main.nonce!);

  const feature = buildFor(run, FEATURE_LIB);
  expect(within(feature.target, chosen)).toBe(true);
  const featureBind = bindgenFor(run, FEATURE_LIB);
  expect(featureBind.input).toBe(join(feature.target, TRIPLE, "release", `${FEATURE_LIB}.wasm`));
  expect(featureBind.content).toBe(feature.nonce!);
  // Separate feature-crate artifact: never the main crate's file.
  expect(featureBind.input).not.toBe(mainBind.input);
}

test("default target: builds and binds under <repo>/target with unchanged names and outputs", () => {
  const box = sandbox();
  const run = runScript(box);
  expect(run.status, run.output).toBe(0);
  expectUnchangedInterface(run);
  expectBoundToTarget(run, join(box.root, "target"));
  expect(bindgenFor(run, MAIN_LIB).input).toBe(join(box.root, "target", TRIPLE, "release", `${MAIN_LIB}.wasm`));
});

test("default target: WASM_FEATURES is forwarded to the main build only", () => {
  const box = sandbox();
  const run = runScript(box, { features: "boundary-benchmark" });
  expect(run.status, run.output).toBe(0);
  expectUnchangedInterface(run, "boundary-benchmark");
  expectBoundToTarget(run, join(box.root, "target"));
});

for (const kind of ["absolute", "relative"] as const) {
  test(`external ${kind} CARGO_TARGET_DIR: bindgen gets this run's artifacts, never stale root output`, () => {
    const box = sandbox();
    const sentinels = placeStaleRootSentinels(box.root);
    const chosen = join(box.cwd, "ext-target");
    const run = runScript(box, {
      targetDir: kind === "absolute" ? chosen : "ext-target",
      features: "boundary-benchmark",
    });
    expect(run.status, run.output).toBe(0);
    expectUnchangedInterface(run, "boundary-benchmark");
    expectBoundToTarget(run, chosen);

    for (const call of bindgens(run)) {
      expect(sentinels).not.toContain(call.input);
      expect(call.content?.startsWith("STALE-SENTINEL")).toBe(false);
    }
    // No build output escapes into the repository target or feature crate dir.
    for (const call of builds(run)) {
      expect(within(call.target, join(box.root, "target"))).toBe(false);
      expect(within(call.target, chosen)).toBe(true);
    }
    expect(listFiles(join(box.root, "target")).sort()).toEqual(sentinels.slice(0, 2).sort());
    expect(listFiles(join(box.root, FEATURE_CRATE, "target"))).toEqual([sentinels[2]!]);
    for (const file of sentinels) expect(readFileSync(file, "utf8")).toBe(`STALE-SENTINEL:${file}`);
  });
}

for (const lib of [MAIN_LIB, FEATURE_LIB]) {
  test(`external CARGO_TARGET_DIR: missing ${lib} artifact fails instead of using another file`, () => {
    const box = sandbox();
    const sentinels = placeStaleRootSentinels(box.root);
    const run = runScript(box, { targetDir: join(box.cwd, "ext-target"), omit: [lib] });
    expect(run.status).not.toBe(0);
    for (const call of bindgens(run)) {
      expect(sentinels).not.toContain(call.input);
      if (argAfter(call.args, "--out-name") === lib) expect(call.content).toBeNull();
    }
  });

  test(`default target: missing ${lib} artifact fails the script`, () => {
    const box = sandbox();
    const run = runScript(box, { omit: [lib] });
    expect(run.status).not.toBe(0);
    for (const call of bindgens(run)) {
      if (argAfter(call.args, "--out-name") === lib) expect(call.content).toBeNull();
    }
  });
}
