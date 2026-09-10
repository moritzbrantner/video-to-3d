#!/usr/bin/env bash
set -euo pipefail

case_name="${GOLDEN_CASE:-lateral}"
exec bash tools/run_colmap_golden.sh \
  "target/golden-reference/$case_name/images" \
  "target/golden-reference/$case_name/colmap-runtime" \
  "$case_name"
