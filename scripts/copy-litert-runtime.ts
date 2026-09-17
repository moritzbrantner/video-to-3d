import { copyFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dir, "..");
const sourceDirectory = join(root, "node_modules", "@litertjs", "core", "wasm");
const targetDirectory = join(root, "apps", "web", "public", "litert-wasm");
const variants = [
  "litert_wasm_internal",
  "litert_wasm_threaded_internal",
  "litert_wasm_compat_internal",
];

mkdirSync(targetDirectory, { recursive: true });
for (const variant of variants) {
  for (const extension of ["js", "wasm"]) {
    copyFileSync(
      join(sourceDirectory, `${variant}.${extension}`),
      join(targetDirectory, `${variant}.${extension}`),
    );
  }
}
