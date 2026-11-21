#!/bin/sh
set -eu
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo 'Usage: run.sh KEYFRAMES_JSON REGISTERED_JSON COLMAP_WORKSPACE OUTPUT_GLB [INVENTORY_JSON]' >&2
  exit 2
fi
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH='' cd -- "$here/../.." && pwd)
work=$3
model="$work/$(cat "$work/selected-model.txt")"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
uv run --script "$here/reconstruct.py" "$work/images" "$model" "$scratch/stereo" --max-image-size 360 >&2
uv run --script "$here/export_grids.py" "$2" "$model" "$scratch/stereo" "$scratch/grids.json"
keyframes=$1
registered=$2
output=$4
shift 4
if cargo run --quiet --release --manifest-path "$root/Cargo.toml" -p video-to-3d-cli -- colmap-surface "$keyframes" "$registered" "$scratch/grids.json" "$output" "$@" >"$scratch/report.json"; then
  cat "$scratch/report.json"
else
  if [ -s "$scratch/report.json" ]; then
    python3 -c 'import json, sys; print(json.load(sys.stdin).get("error", "Rust rejected dense evidence"), file=sys.stderr)' <"$scratch/report.json"
  fi
  exit 1
fi
