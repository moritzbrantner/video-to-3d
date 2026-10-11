"""Derive the Blender import expectation of a synthetic scene fixture (issue #151).

    python3 tools/blender/expected_from_synthetic_scene.py <fixture dir> <expected.json>

The expectation is taken from the fixture's independently generated ground truth
(`truth/scene.json`), never from an exporter's output: every architectural element
and object must arrive as its own selectable mesh object, with the fixture's
minimum number of independent objects, and its world bounds in Blender
coordinates (x right, y forward, z up) within an authored tolerance.
"""

import json
import sys
from pathlib import Path

SCHEMA = "video-to-3d/blender-scene-expectation/v1"
# Authored for the exact synthetic handoff: the fixture's float32 PLY vertices and
# six-decimal transforms reproduce bounds to well under a millimetre.
SYNTHETIC_BOUNDS_TOLERANCE_M = 0.001


def blender_bounds(world_bounds):
    """Fixture world (x right, y down, z forward) to Blender (x, z, -y)."""
    (x0, y0, z0), (x1, y1, z1) = world_bounds
    return [[x0, z0, -y1], [x1, z1, -y0]]


def expectation(scene):
    inventory = scene["expected_inventory"]
    nodes = {node["id"]: node for node in scene["nodes"]}
    return {
        "schema": SCHEMA,
        "source": f"{scene['schema']} case {scene['case']}",
        "objects": inventory["objects"],
        "architecture": inventory["architecture"],
        "min_independent_object_nodes": inventory["min_independent_object_nodes"],
        "labels": {node_id: node["label"] for node_id, node in nodes.items()},
        "bounds_blender_m": {
            node_id: blender_bounds(node["world_bounds_m"]) for node_id, node in nodes.items()
        },
        "bounds_tolerance_m": SYNTHETIC_BOUNDS_TOLERANCE_M,
        "edit_translation_m": [0.5, 0.0, 0.0],
    }


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: expected_from_synthetic_scene.py <fixture dir> <expected.json>")
    scene = json.loads((Path(argv[1]) / "truth" / "scene.json").read_text())
    Path(argv[2]).write_text(json.dumps(expectation(scene), indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv)
