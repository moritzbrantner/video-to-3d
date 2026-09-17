import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dir, "..");
const sourceCandidates = [
  join(root, "apps", "web", "node_modules", "@litertjs", "core", "wasm"),
  join(root, "node_modules", "@litertjs", "core", "wasm"),
];
const sourceDirectory = sourceCandidates.find((candidate) => existsSync(candidate));
if (!sourceDirectory) {
  throw new Error(
    `LiteRT.js runtime directory was not installed; checked: ${sourceCandidates.join(", ")}`,
  );
}

const targetDirectory = join(root, "apps", "web", "public", "litert-wasm");
const variants = [
  "litert_wasm_internal",
  "litert_wasm_threaded_internal",
  "litert_wasm_compat_internal",
];

mkdirSync(targetDirectory, { recursive: true });
for (const variant of variants) {
  for (const extension of ["js", "wasm"]) {
    const source = join(sourceDirectory, `${variant}.${extension}`);
    if (!existsSync(source)) {
      throw new Error(`LiteRT.js runtime asset is missing: ${source}`);
    }
    copyFileSync(source, join(targetDirectory, `${variant}.${extension}`));
  }
}
