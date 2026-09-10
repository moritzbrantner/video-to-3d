#!/usr/bin/env bash
set -euo pipefail

implementation=${VIDEO_TO_3D_IMPLEMENTATION:?VIDEO_TO_3D_IMPLEMENTATION must be ours or colmap}
fixture=${VIDEO_TO_3D_FIXTURE:?VIDEO_TO_3D_FIXTURE must point to the generated golden scene}
output=$(mktemp)
trap 'rm -f "$output"' EXIT

case "$implementation" in
  ours)
    target/release/examples/golden_reconstruct "$fixture" "$output" >/dev/null
    ;;
  colmap)
    scripts/run_colmap_reference.sh "$fixture" "$output" >/dev/null
    ;;
  *)
    echo "unsupported VIDEO_TO_3D_IMPLEMENTATION=$implementation" >&2
    exit 2
    ;;
esac

test -s "$output"
