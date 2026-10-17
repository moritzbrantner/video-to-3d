"""Synthetic resource inputs for #84; never a production scene exporter.

Read unchanged #75 truth meshes. Blender encodes each local mesh as a resource;
the product must subsequently export the canonical scene, identity and placement.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).parent))
from expected_from_synthetic_scene import expectation


from build_truth_scene import read_ply


def content_hash(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def transform(matrix):
    position, rotation, scale = matrix.decompose()
    return {"translation": list(position), "rotation": [rotation.x, rotation.y, rotation.z, rotation.w], "scale": list(scale)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    fixture, root = Path(args.fixture), Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "resources").mkdir(exist_ok=True)
    (root / "sampled").mkdir(exist_ok=True)
    for image in (fixture / "images").glob("*.ppm"):
        shutil.copyfile(image, root / "sampled" / image.name)
    truth = json.loads((fixture / "truth/scene.json").read_text())
    # Canonical scene is RH/Y-up, while #75 truth is camera-frame Y-down/Z-forward.
    canonical_from_truth = Matrix.Diagonal((1, -1, -1, 1))
    # Blender's GLB export converts Blender Z-up to GLB Y-up. Feed local vertices
    # in Blender coordinates so the resource payload's positions are (x,-y,-z).
    blender_from_truth = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
    group = Matrix.Translation(Vector((0.25, 0.5, -0.75))) @ Quaternion((0, 1, 0), math.radians(22)).to_matrix().to_4x4()
    scene = {"schema_version": 1, "frame": {"unit": "meters", "handedness": "right_handed", "up_axis": "positive_y"}, "resources": [], "materials": [], "assets": []}
    scene["assets"].append({"id": "room-group", "role": "environment", "transform": transform(group), "collision": {"box": {"half_extents": [0.05, 0.05, 0.05]}}, "collision_provenance": "authored"})
    bpy.ops.wm.read_factory_settings(use_empty=True)
    texture = bpy.data.images.new("acceptance-checker", width=2, height=2)
    texture.pixels = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]
    texture.filepath_raw = str(root / "resources/checker.png")
    texture.file_format = "PNG"
    texture.save()
    scene["resources"].append({"id": "checker", "kind": "texture", "path": "resources/checker.png", "content_hash": content_hash(root / "resources/checker.png"), "provenance": ["authored"]})
    for node in truth["nodes"]:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        positions, triangles = read_ply(fixture / node["mesh"])
        mesh = bpy.data.meshes.new(node["id"])
        local_positions = [tuple((blender_from_truth @ Vector((*point, 1)))[:3]) for point in positions]
        mesh.from_pydata(local_positions, [], triangles)
        mesh.validate()
        # A real UV layout; canonical material/texture must survive downstream.
        uv = mesh.uv_layers.new(name="acceptance-uv")
        for polygon in mesh.polygons:
            for corner, loop in enumerate(polygon.loop_indices):
                uv.data[loop].uv = ((corner % 2), (corner // 2))
        obj = bpy.data.objects.new("resource-local", mesh)
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        path = root / f"resources/{node['id']}.glb"
        bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True)
        resource = {"id": f"{node['id']}-mesh", "kind": "mesh", "path": f"resources/{node['id']}.glb", "content_hash": content_hash(path), "provenance": ["authored"], "confidence": 0.75}
        scene["resources"].append(resource)
        material = {"id": f"{node['id']}-material", "base_color": [*node["albedo"], 1], "base_color_texture": "checker", "metallic": 0.0, "roughness": 0.8, "provenance": "authored"}
        scene["materials"].append(material)
        m = node["world_from_local"]
        world = Matrix([m[:4], m[4:8], m[8:12], m[12:]])
        canonical_world = canonical_from_truth @ world @ canonical_from_truth
        scene["assets"].append({"id": node["id"], "role": "editable_object" if node["kind"] == "object" else "environment", "parent": "room-group", "transform": transform(group.inverted() @ canonical_world), "visual": {"mesh": {"mesh": resource["id"], "material": material["id"]}}, "labels": [{"name": node["label"], "confidence": 0.8}]})
    # Deliberately labeled completion sentinel reuses real crate geometry. It is
    # not an inferred object or a change to the independently authored inventory.
    crate = next(resource for resource in scene["resources"] if resource["id"] == "crate-mesh")
    shutil.copyfile(root / crate["path"], root / "resources/completion.glb")
    scene["resources"].append({**crate, "id": "completion-mesh", "path": "resources/completion.glb", "provenance": ["generative_completion"]})
    scene["assets"].append({"id": "completion-sentinel", "role": "editable_object", "transform": {"translation": [8, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]}, "visual": {"mesh": {"mesh": "completion-mesh", "material": "crate-material"}}, "labels": [{"name": "synthetic completion sentinel"}]})
    # Real mesh collision reference alongside the authored primitive group proxy.
    floor = next(asset for asset in scene["assets"] if asset["id"] == "floor")
    floor["collision"] = {"mesh": {"mesh": "floor-mesh"}}
    (root / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    expected = expectation(truth)
    expected["expected_parent_ids"] = {node["id"]: "room-group" for node in truth["nodes"]}
    (root / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    (root / "provenance.txt").write_text("Synthetic #75 exact-truth mesh resources, not reconstructed or filmed product output.\n")


main()
