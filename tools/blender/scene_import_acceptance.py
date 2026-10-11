"""Headless Blender import/edit acceptance for a scene handoff GLB (issue #151).

Run inside a pinned Blender:

    blender --background --factory-startup --python tools/blender/scene_import_acceptance.py -- \
        --glb scene.glb --expected expected.json --report report.json [--render render.png]

The expectation (`video-to-3d/blender-scene-expectation/v1`) is authored
independently of the exporter, e.g. by `expected_from_synthetic_scene.py` from a
fixture's ground truth. Objects are matched by the `video_to_3d_id` custom property
(glTF node extras) or, failing that, by object name.

Checks, each reported individually; the process exits non-zero unless all pass:

- the GLB imports and every expected architectural element and object maps to
  exactly one mesh object;
- at least `min_independent_object_nodes` expected objects are separate,
  selectable mesh objects with their own mesh data (one mesh with disconnected
  islands is a single node and does not count);
- every matched mesh has polygons, non-degenerate bounds, a finite transform and a
  material;
- world bounds agree with the expectation within `bounds_tolerance_m` when given;
- moving one object by `edit_translation_m` moves exactly its bounds by that
  amount and leaves every other object's transform and world-space vertices
  unchanged.

The report also lists the imported hierarchy (names, parents, mesh data,
materials, custom properties) and, with `--render`, writes a canonical render.
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Vector

EXPECTATION_SCHEMA = "video-to-3d/blender-scene-expectation/v1"
REPORT_SCHEMA = "video-to-3d/blender-scene-acceptance/v1"


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--render")
    return parser.parse_args(argv)


def world_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return [[min(c[axis] for c in corners) for axis in range(3)], [max(c[axis] for c in corners) for axis in range(3)]]


def world_vertex_digest(obj):
    digest = hashlib.sha256()
    for vertex in obj.data.vertices:
        world = obj.matrix_world @ vertex.co
        digest.update(struct.pack("<3d", *world))
    return digest.hexdigest()


def object_id(obj):
    return str(obj.get("video_to_3d_id", obj.name))


def custom_properties(obj):
    return {key: str(obj[key]) for key in obj.keys() if not key.startswith("_")}


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=""):
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})
        return passed

    @property
    def passed(self):
        return bool(self.checks) and all(check["passed"] for check in self.checks)


def hierarchy():
    return [
        {
            "name": obj.name,
            "id": object_id(obj),
            "type": obj.type,
            "parent": obj.parent.name if obj.parent else None,
            "mesh": obj.data.name if obj.type == "MESH" else None,
            "vertices": len(obj.data.vertices) if obj.type == "MESH" else 0,
            "polygons": len(obj.data.polygons) if obj.type == "MESH" else 0,
            "materials": [slot.material.name for slot in obj.material_slots if slot.material]
            if obj.type == "MESH"
            else [],
            "world_bounds_m": world_bounds(obj) if obj.type == "MESH" else None,
            "custom_properties": custom_properties(obj),
        }
        for obj in sorted(bpy.data.objects, key=lambda obj: obj.name)
    ]


def render(path, meshes):
    scene = bpy.context.scene
    low = Vector([min(world_bounds(obj)[0][axis] for obj in meshes) for axis in range(3)])
    high = Vector([max(world_bounds(obj)[1][axis] for obj in meshes) for axis in range(3)])
    center = (low + high) / 2
    size = high - low
    camera_data = bpy.data.cameras.new("acceptance-camera")
    camera_data.lens = 18.0
    camera = bpy.data.objects.new("acceptance-camera", camera_data)
    scene.collection.objects.link(camera)
    # Canonical view from inside the scene bounds (an enclosed room hides its
    # contents from any outside view): near the front (-y), raised, looking at the
    # center of the lower half.
    camera.location = Vector((center.x, low.y + 0.08 * size.y, low.z + 0.6 * size.z))
    target = Vector((center.x, center.y + 0.15 * size.y, low.z + 0.2 * size.z))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    # A point light at the camera: a sun cannot reach into an enclosed room.
    light_data = bpy.data.lights.new("acceptance-light", "POINT")
    light_data.energy = 6.0 * max(size.length, 1.0) ** 2
    light = bpy.data.objects.new("acceptance-light", light_data)
    light.location = camera.location
    scene.collection.objects.link(light)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.render.resolution_x, scene.render.resolution_y = 640, 360
    scene.render.filepath = str(Path(path).resolve())
    bpy.ops.render.render(write_still=True)


def main():
    args = arguments()
    expected = json.loads(Path(args.expected).read_text())
    report = Report()
    output = {"schema": REPORT_SCHEMA, "glb": args.glb, "blender": bpy.app.version_string}
    try:
        if expected.get("schema") != EXPECTATION_SCHEMA:
            raise ValueError(f"unsupported expectation schema {expected.get('schema')!r}")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            result = bpy.ops.import_scene.gltf(filepath=str(Path(args.glb).resolve()))
            imported = result == {"FINISHED"}
            detail = "" if imported else str(result)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            imported, detail = False, str(error)
        if not report.check("glb_imports", imported, detail):
            return finish(report, output, args)
        output["hierarchy"] = hierarchy()

        meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
        by_id = {}
        for obj in meshes:
            by_id.setdefault(object_id(obj), []).append(obj)
        matched = {}
        for kind in ("architecture", "objects"):
            for element in expected[kind]:
                candidates = by_id.get(element, [])
                if report.check(
                    f"{kind}:{element}:single_mesh_node",
                    len(candidates) == 1,
                    f"{len(candidates)} mesh object(s) carry this id",
                ):
                    matched[element] = candidates[0]

        objects = [matched[element] for element in expected["objects"] if element in matched]
        own_data = {obj.data.name for obj in objects}
        independent = [obj for obj in objects if obj.data.users == 1 or len(own_data) == len(objects)]
        report.check(
            "independent_object_nodes",
            len(objects) >= expected["min_independent_object_nodes"]
            and len(own_data) == len(objects)
            and len(independent) == len(objects),
            f"{len(objects)} expected objects matched to {len(own_data)} distinct meshes; "
            f"at least {expected['min_independent_object_nodes']} required",
        )

        tolerance = expected.get("bounds_tolerance_m")
        for element, obj in sorted(matched.items()):
            low, high = world_bounds(obj)
            extents = sorted(high[axis] - low[axis] for axis in range(3))
            finite = all(math.isfinite(value) for row in obj.matrix_world for value in row)
            report.check(
                f"{element}:geometry",
                len(obj.data.polygons) > 0 and extents[1] > 1e-4 and finite,
                f"{len(obj.data.polygons)} polygons, extents {extents}",
            )
            report.check(
                f"{element}:material",
                any(slot.material for slot in obj.material_slots),
                ", ".join(slot.material.name for slot in obj.material_slots if slot.material),
            )
            truth = expected.get("bounds_blender_m", {}).get(element)
            if truth is not None and tolerance is not None:
                error = max(abs(a - b) for got, want in zip((low, high), truth) for a, b in zip(got, want))
                report.check(f"{element}:bounds", error <= tolerance, f"max bounds error {error:.6f} m")

        if objects:
            moved = objects[0]
            delta = Vector(expected.get("edit_translation_m", [0.5, 0.0, 0.0]))
            others = [obj for obj in meshes if obj != moved]
            before = {obj.name: (obj.matrix_world.copy(), world_vertex_digest(obj)) for obj in others}
            low_before = Vector(world_bounds(moved)[0])
            moved.location = moved.location + delta
            bpy.context.view_layer.update()
            shift = Vector(world_bounds(moved)[0]) - low_before
            untouched = all(
                obj.matrix_world == before[obj.name][0] and world_vertex_digest(obj) == before[obj.name][1]
                for obj in others
            )
            report.check(
                "edit_moves_only_one_object",
                (shift - delta).length < 1e-6 and untouched,
                f"moved {object_id(moved)} by {list(shift)}; other objects unchanged: {untouched}",
            )
            moved.location = moved.location - delta
            bpy.context.view_layer.update()

        if args.render and meshes:
            render(args.render, meshes)
            output["render"] = args.render
    except Exception as error:  # noqa: BLE001 - every failure is a failed report
        report.check("harness", False, f"{type(error).__name__}: {error}")
    return finish(report, output, args)


def finish(report, output, args):
    output["checks"] = report.checks
    output["passed"] = report.passed
    Path(args.report).write_text(json.dumps(output, indent=2) + "\n")
    failed = [check["check"] for check in report.checks if not check["passed"]]
    print(f"blender-scene-acceptance passed={report.passed} checks={len(report.checks)} failed={failed}")
    return 0 if report.passed else 1


sys.exit(main())
