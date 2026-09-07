#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/apps/web/public/wasm"
mkdir -p "$OUT"

cargo build \
  --manifest-path "$ROOT/Cargo.toml" \
  --release \
  --target wasm32-unknown-unknown \
  -p video-to-3d-wasm

wasm-bindgen \
  --target web \
  --out-dir "$OUT" \
  --out-name video_to_3d_wasm \
  "$ROOT/target/wasm32-unknown-unknown/release/video_to_3d_wasm.wasm"
