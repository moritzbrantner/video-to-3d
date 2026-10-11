"""Build a ground-truth Blender handoff GLB from a synthetic scene fixture (issue #151).

Run inside Blender:

    blender --background --factory-startup --python tools/blender/build_truth_scene.py -- \
        --fixture <synthetic_room_fixture dir> --out truth.glb [--variant truth|joined|missing-object]

The `truth` variant exports every scene element of `truth/scene.json` as its own
named mesh object (local PLY mesh, `world_from_local` transform, one material),
which is what a correct production export must look like. The other variants are
deliberately broken handoffs the acceptance check must reject:

- `joined`: all objects merged into one mesh with disconnected islands;
- `missing-object`: the first object is left out;
- `parented`: the third object is parented to the second (world placement kept),
  so moving the second moves both;
- `architecture-parented`: the last object is parented to the floor (world
  placement kept), so moving the floor moves the object;
- `shared-data`: an extra, unidentified object reuses the last object's mesh data.

This script only produces reference inputs for the acceptance harness
(`scene_import_acceptance.py`); it is not the product exporter.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

# The fixture world is x right, y down, z forward; Blender is x right, y forward, z up.
BLENDER_FROM_WORLD = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def read_ply(path):
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    vertices = int(next(line.split()[2] for line in header if line.startswith("element vertex")))
    faces = int(next(line.split()[2] for line in header if line.startswith("element face")))
    positions = [struct.unpack_from("<3f", data, end + 12 * index) for index in range(vertices)]
    start = end + 12 * vertices
    triangles = []
    for index in range(faces):
        offset = start + 13 * index
        assert data[offset] == 3, "triangles only"
        triangles.append(struct.unpack_from("<3I", data, offset + 1))
    return positions, triangles


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", choices=["truth", "joined", "missing-object", "parented", "architecture-parented", "shared-data"], default="truth")
    return parser.parse_args(argv)


def main():
    args = arguments()
    fixture = Path(args.fixture)
    scene = json.loads((fixture / "truth" / "scene.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objects = []
    skipped = False
    for node in scene["nodes"]:
        if args.variant == "missing-object" and node["kind"] == "object" and not skipped:
            skipped = True
            continue
        positions, triangles = read_ply(fixture / node["mesh"])
        mesh = bpy.data.meshes.new(node["id"])
        mesh.from_pydata(positions, [], triangles)
        mesh.validate()
        material = bpy.data.materials.new(f"{node['id']}-material")
        material.diffuse_color = (*node["albedo"], 1.0)
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is not None:
            principled.inputs["Base Color"].default_value = (*node["albedo"], 1.0)
        mesh.materials.append(material)
        obj = bpy.data.objects.new(node["id"], mesh)
        m = node["world_from_local"]
        world_from_local = Matrix([m[0:4], m[4:8], m[8:12], m[12:16]])
        obj.matrix_world = BLENDER_FROM_WORLD @ world_from_local
        obj["video_to_3d_id"] = node["id"]
        obj["video_to_3d_kind"] = node["kind"]
        obj["video_to_3d_label"] = node["label"]
        bpy.context.scene.collection.objects.link(obj)
        objects.append((node, obj))
    movable = [obj for node, obj in objects if node["kind"] == "object"]
    if args.variant == "parented":
        parent, child = movable[1], movable[2]
        world = child.matrix_world.copy()
        child.parent = parent
        child.matrix_world = world
    if args.variant == "architecture-parented":
        floor = next(obj for node, obj in objects if node["id"] == "floor")
        child = movable[-1]
        world = child.matrix_world.copy()
        child.parent = floor
        child.matrix_world = world
    if args.variant == "shared-data":
        decoy = bpy.data.objects.new("decoy", movable[-1].data)
        decoy.matrix_world = movable[-1].matrix_world.copy()
        decoy.location.x += 10.0
        bpy.context.scene.collection.objects.link(decoy)
    if args.variant == "joined":
        movable = [obj for node, obj in objects if node["kind"] == "object"]
        bpy.ops.object.select_all(action="DESELECT")
        for obj in movable:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = movable[0]
        bpy.ops.object.join()
        movable[0].name = "objects"
        movable[0]["video_to_3d_id"] = "objects"
    bpy.ops.export_scene.gltf(filepath=str(Path(args.out).resolve()), export_format="GLB", export_extras=True)


main()
