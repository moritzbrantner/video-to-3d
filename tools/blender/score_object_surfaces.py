"""Score exported object surfaces against named CAD instances in a Blender capture.

Run with blender -b --python-exit-code 1 -P tools/blender/score_object_surfaces.py --
  --source input.blend --cameras cameras.json --glb scene.glb --pairs pairs.json --report score.json

pairs.json is an array of {"object_id":"...", "source_instance":"chair.016",
"exclude_parts":["Plane.003"]}. Use independently established correspondences.
Both scores use independently seeded, area-weighted sampling of each complete
surface and exact nearest triangle distances in the source scene's length unit.
camera_only keeps the exported object pose after the global camera calibration.
pose_aligned estimates a per-object rigid transform by multi-start symmetric ICP.
This estimate preserves size and chirality and can converge to a local minimum.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from object_surface_metrics import Surface, move, rigid_fit, surface_scores
from score_rendered_scene import align


def mesh_arrays(mesh, matrix):
    mesh.calc_loop_triangles()
    positions = np.empty(len(mesh.vertices) * 3, np.float32)
    indices = np.empty(len(mesh.loop_triangles) * 3, np.int32)
    mesh.vertices.foreach_get("co", positions)
    mesh.loop_triangles.foreach_get("vertices", indices)
    return move(positions.reshape(-1, 3), np.array(matrix)), indices.reshape(-1, 3)


def load_objects(path, poses):
    with path.open("rb") as file:
        magic, version, _, length, kind = struct.unpack("<4s4I", file.read(20))
        if (magic, version, kind) != (b"glTF", 2, 0x4E4F534A):
            raise ValueError("expected a glTF 2.0 binary")
        doc = json.loads(file.read(length))
    cameras = doc["asset"]["extras"]["video_to_3d"]["accepted_camera_evidence"]
    transform = align(cameras, poses)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    # The editable GLB can also contain diagnostic scenes. Score active scene roots only.
    nodes = [doc["nodes"][i] for i in doc["scenes"][doc.get("scene", 0)]["nodes"]]
    objects = {}
    for node in nodes:
        meta = node.get("extras", {}).get("video_to_3d", {})
        if meta.get("role") != "completed_object_surface":
            continue
        key = meta["object_id"]
        if key in objects:
            raise ValueError(f"duplicate completed mesh for {key}")
        obj = bpy.data.objects[node["name"]]
        vertices, faces = mesh_arrays(obj.data, obj.matrix_world)
        objects[key] = (meta["method"], move(vertices, transform), faces)
    return objects, transform


def source_surfaces(path, capture, pairs):
    bpy.ops.wm.open_mainfile(filepath=str(path.resolve()), load_ui=False, use_scripts=False)
    if "scene" in capture:
        bpy.context.window.scene = bpy.data.scenes[capture["scene"]]
    bpy.context.scene.frame_set(capture.get("scene_frame", 1))
    parts = {key: [] for key in pairs}
    for inst in bpy.context.evaluated_depsgraph_get().object_instances:
        obj = inst.object
        if obj.type != "MESH" or obj.hide_render or not obj.visible_camera:
            continue
        name = inst.parent.name if inst.parent else obj.name
        for key, pair in pairs.items():
            if name != pair["source_instance"] or obj.name in pair.get("exclude_parts", []):
                continue
            mesh = obj.to_mesh()
            try:
                parts[key].append(mesh_arrays(mesh, inst.matrix_world))
            finally:
                obj.to_mesh_clear()
    result = {}
    for key, meshes in parts.items():
        if not meshes:
            raise ValueError(f"no CAD meshes for {pairs[key]['source_instance']!r}")
        vertices, faces, count = [], [], 0
        for v, f in meshes:
            vertices.append(v)
            faces.append(f + count)
            count += len(v)
        result[key] = Surface(np.vstack(vertices), np.vstack(faces))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("source", "cameras", "glb", "pairs", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000, help="score samples per direction")
    parser.add_argument("--threshold", type=float, default=.10, help="tolerance in capture units")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.samples < 1 or not np.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("samples and threshold must be positive")
    capture = json.loads(args.cameras.read_text())
    requested = json.loads(args.pairs.read_text())
    pairs = {p["object_id"]: p for p in requested}
    if not pairs or len(pairs) != len(requested):
        parser.error("pairs must list unique object IDs")
    objects, transform = load_objects(args.glb, np.array([f["matrix_world"] for f in capture["frames"]]))
    reference = source_surfaces(args.source, capture, pairs)
    report = {"schema_version": 1, "glb": args.glb.name, "source": args.source.name,
              "unit": capture.get("unit", "source_scene_unit"), "threshold": args.threshold,
              "samples_per_direction": args.samples, "camera_only_similarity": transform.tolist(),
              "pose_fit": "multi_start_symmetric_icp_rigid_no_scale",
              "distance": "bidirectional_area_weighted_sample_to_nearest_triangle",
              "completed_objects_without_correspondence": sorted(objects.keys() - pairs.keys()),
              "objects": []}
    for key, pair in pairs.items():
        row = dict(pair)
        report["objects"].append(row)
        if key not in objects:
            row["status"] = "missing_completed_object"
            continue
        method, vertices, faces = objects[key]
        mesh, cad = Surface(vertices, faces), reference[key]
        pose = rigid_fit(mesh, cad)
        row.update(status="scored", method=method, triangles=len(faces),
                   camera_only=surface_scores(mesh, cad, np.eye(4), args.samples, args.threshold),
                   pose_aligned=surface_scores(mesh, cad, pose, args.samples, args.threshold),
                   evaluation_only_rigid_transform=pose.tolist())
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Scored", sum(r["status"] == "scored" for r in report["objects"]), "of", len(pairs), "objects")


if __name__ == "__main__":
    main()
