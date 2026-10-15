#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/apps/web/public/wasm"
FEATURE_OUT="$ROOT/apps/web/public/feature-wasm"
mkdir -p "$OUT" "$FEATURE_OUT"

# Resolve the target directory Cargo will actually use (CARGO_TARGET_DIR,
# CARGO_BUILD_TARGET_DIR, cargo config or the default), so wasm-bindgen consumes the artifact
# this build produced rather than a possibly stale one under $ROOT/target (#192).
TARGET="$(
  cargo metadata --format-version 1 --no-deps --manifest-path "$ROOT/Cargo.toml" |
    sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p'
)"
if [[ -z "$TARGET" ]]; then
  echo "build-wasm: could not resolve the Cargo target directory" >&2
  exit 1
fi
# The feature crate keeps its own artifacts, inside the same chosen target root.
FEATURE_TARGET="$TARGET/feature-lab"

require_artifact() {
  if [[ ! -f "$1" ]]; then
    echo "build-wasm: expected freshly built artifact is missing: $1" >&2
    exit 1
  fi
}

FEATURE_ARGS=()
if [[ -n "${WASM_FEATURES:-}" ]]; then
  FEATURE_ARGS=(--features "$WASM_FEATURES")
fi

cargo build \
  --manifest-path "$ROOT/Cargo.toml" \
  --release \
  --target wasm32-unknown-unknown \
  --target-dir "$TARGET" \
  "${FEATURE_ARGS[@]}" \
  -p video-to-3d-wasm

MAIN_WASM="$TARGET/wasm32-unknown-unknown/release/video_to_3d_wasm.wasm"
require_artifact "$MAIN_WASM"
wasm-bindgen \
  --target web \
  --out-dir "$OUT" \
  --out-name video_to_3d_wasm \
  "$MAIN_WASM"

cargo build \
  --manifest-path "$ROOT/crates/video-to-3d-feature-wasm/Cargo.toml" \
  --release \
  --target wasm32-unknown-unknown \
  --target-dir "$FEATURE_TARGET"

FEATURE_WASM="$FEATURE_TARGET/wasm32-unknown-unknown/release/video_to_3d_feature_wasm.wasm"
require_artifact "$FEATURE_WASM"
wasm-bindgen \
  --target web \
  --out-dir "$FEATURE_OUT" \
  --out-name video_to_3d_feature_wasm \
  "$FEATURE_WASM"
