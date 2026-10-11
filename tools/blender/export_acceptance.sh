#!/usr/bin/env bash
# #84 production-export acceptance; all scene/resource inputs are synthetic truth.
set -euo pipefail
out=$(realpath -m "${1:?usage: export_acceptance.sh <fresh output dir> [blender]}")
blender=${2:-blender}
repo=$(cd "$(dirname "$0")/../.." && pwd)
[ ! -e "$out" ] || { echo "output directory must be fresh" >&2; exit 1; }
mkdir -p "$out"
cd "$repo"
cargo run --quiet --release --locked -p video-to-3d-core --example synthetic_room_fixture -- "$out/truth" controlled-room
"$blender" --background --factory-startup --python "$repo/tools/blender/prepare_export_acceptance.py" -- --fixture "$out/truth" --out "$out/project-inputs"
# Uses the actual product CLI, records synthetic imported canonical scene through
# ProjectStore, exports its bundle and removes the original project workspace.
VIDEO_TO_3D_EXPORT_FIXTURE="$out/project-inputs" VIDEO_TO_3D_EXPORT_PORTABLE="$out/portable" \
  cargo test --release --locked -p video-to-3d-cli --test scene_export -- --ignored
"$blender" --background --factory-startup --python "$repo/tools/blender/scene_import_acceptance.py" -- \
  --glb "$out/portable/scene.glb" --expected "$out/project-inputs/expected.json" \
  --report "$out/production-export-report.json" --render "$out/production-export.png"
"$blender" --background --factory-startup --python "$repo/tools/blender/check_export_materials.py" -- \
  --glb "$out/portable/scene.glb" --inputs "$out/project-inputs" --report "$out/material-export-report.json"
