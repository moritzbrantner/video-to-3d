#!/usr/bin/env bash
# Self-test of the headless Blender scene acceptance (issue #151).
#
#   tools/blender/self_test.sh <output dir> [blender binary]
#
# Generates the synthetic room fixture (#75), derives its independent Blender
# expectation, builds a correct ground-truth handoff GLB and five broken ones
# (joined islands, missing object, object- and architecture-parented objects,
# shared mesh data), and
# requires the harness to pass the correct handoff and to reject each broken one
# for exactly the expected checks. Reports and the canonical render of the correct
# handoff are left in the output directory.
set -euo pipefail

out=$(realpath -m "${1:?usage: self_test.sh <output dir> [blender]}")
blender=${2:-blender}
repo=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out"

cargo run --quiet --release --locked -p video-to-3d-core --example synthetic_room_fixture -- \
  "$out/fixture" controlled-room
python3 "$repo/tools/blender/expected_from_synthetic_scene.py" "$out/fixture" "$out/expected.json"

run_blender() {
  "$blender" --background --factory-startup --python-use-system-env --python "$@"
}

for variant in truth joined missing-object parented architecture-parented shared-data; do
  run_blender "$repo/tools/blender/build_truth_scene.py" -- \
    --fixture "$out/fixture" --out "$out/$variant.glb" --variant "$variant"
  render=()
  if [ "$variant" = truth ]; then render=(--render "$out/render-truth.png"); fi
  set +e
  run_blender "$repo/tools/blender/scene_import_acceptance.py" -- \
    --glb "$out/$variant.glb" --expected "$out/expected.json" \
    --report "$out/report-$variant.json" "${render[@]}"
  status=$?
  set -e
  echo "acceptance $variant exit=$status"
  case "$variant" in
    truth) [ "$status" -eq 0 ] || { echo "correct handoff was rejected" >&2; exit 1; } ;;
    *) [ "$status" -ne 0 ] || { echo "broken handoff ($variant) was accepted" >&2; exit 1; } ;;
  esac
done

# Optional fixture-owned hierarchy accepts legitimate grouping while strict
# defaults above continue to reject accidental coupled object parenting.
python3 - "$out" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
expected = json.loads((out / "expected.json").read_text())
expected["expected_parent_ids"] = {node: "room-group" for node in expected["architecture"] + expected["objects"]}
(out / "expected-grouped.json").write_text(json.dumps(expected))
expected["expected_parent_ids"]["floor"] = None
(out / "expected-wrong-parent.json").write_text(json.dumps(expected))
PY
run_blender "$repo/tools/blender/build_truth_scene.py" -- --fixture "$out/fixture" --out "$out/grouped.glb" --variant grouped
run_blender "$repo/tools/blender/scene_import_acceptance.py" -- --glb "$out/grouped.glb" --expected "$out/expected-grouped.json" --report "$out/report-grouped.json"
set +e
run_blender "$repo/tools/blender/scene_import_acceptance.py" -- --glb "$out/grouped.glb" --expected "$out/expected-wrong-parent.json" --report "$out/report-wrong-parent.json"
wrong_parent_status=$?
set -e
[ "$wrong_parent_status" -ne 0 ] || { echo "wrong hierarchy was accepted" >&2; exit 1; }

python3 - "$out" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
def failed(variant):
    report = json.loads((out / f"report-{variant}.json").read_text())
    return {check["check"] for check in report["checks"] if not check["passed"]}

assert json.loads((out / "report-grouped.json").read_text())["passed"]
assert failed("wrong-parent") == {"floor:parent"}, failed("wrong-parent")
truth = json.loads((out / "report-truth.json").read_text())
assert truth["passed"] and (out / "render-truth.png").stat().st_size > 0
assert {f"edit_isolation:{name}" for name in ("cabinet", "crate", "stool", "ball", "floor", "door")} <= {
    check["check"] for check in truth["checks"]
}
joined = failed("joined")
assert "independent_object_nodes" in joined, joined
assert all(f"objects:{name}:single_mesh_node" in joined for name in ("cabinet", "crate", "stool", "ball")), joined
missing = failed("missing-object")
assert missing == {"objects:cabinet:single_mesh_node"}, missing
# Coupled edits must fail specifically on edit isolation / own mesh data.
# The child also moves in its rotated parent's space, so its own edit is off too.
assert failed("parented") == {"edit_isolation:crate", "edit_isolation:stool", "stool:unparented"}, failed("parented")
assert failed("architecture-parented") == {"edit_isolation:floor", "ball:unparented"}, failed(
    "architecture-parented"
)
assert failed("shared-data") == {"ball:own_mesh_data"}, failed("shared-data")
print("blender scene acceptance self-test passed")
PY
