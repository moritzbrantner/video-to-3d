#!/usr/bin/env bash
set -euo pipefail

case_name="${GOLDEN_CASE:-lateral}"
exec target/release/examples/golden_fixture - "$case_name"
