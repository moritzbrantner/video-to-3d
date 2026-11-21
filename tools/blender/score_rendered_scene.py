"""Compare an exported surface with the source seen by known Blender cameras."""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def load_export(path):
    # Read only the JSON chunk. Blender handles geometry buffers and node transforms.
    with path.open("rb") as stream:
        magic, version, _, length, kind = struct.unpack("<4s4I", stream.read(20))
        if (magic, version, kind) != (b"glTF", 2, 0x4E4F534A):
            raise ValueError("expected a glTF 2.0 binary with a JSON chunk")
        asset = json.loads(stream.read(length))["asset"]["extras"]["video_to_3d"]
    if asset.get("schema") != "video-to-3d/textured-surface":
        raise ValueError("expected a video-to-3d textured surface")
    if "accepted_camera_evidence" not in asset:
        raise ValueError("camera poses are missing; re-export with the current build")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    vertices, faces = [], []
    for surface in bpy.context.scene.objects:
        if surface.type != "MESH":
            continue
        metadata = surface.get("video_to_3d", {})
        if surface.name != "accepted-surface" and metadata.get("role") != "observed_object_surface":
            continue
        start = len(vertices)
        surface.data.calc_loop_triangles()
        vertices.extend(tuple(surface.matrix_world @ v.co) for v in surface.data.vertices)
        faces.extend(tuple(start + index for index in tri.vertices) for tri in surface.data.loop_triangles)
    if not faces:
        raise ValueError("export contains no accepted surface triangles")
    return asset["accepted_camera_evidence"], np.asarray(vertices), faces


def align(cameras, poses):
    """Fit a proper similarity from camera orientations and centers only."""
    indices = [c["frame_index"] for c in cameras]
    if len(set(indices)) != len(indices) or len(indices) < 2:
        raise ValueError("alignment requires at least two distinct registered frames")
    if min(indices) < 0 or max(indices) >= len(poses):
        raise ValueError("registered frame is missing from the reference capture")
    gt = poses[indices]
    rotations = np.array([c["rotation"] for c in cameras]).reshape(-1, 3, 3)
    centers = np.array([-r.T @ c["translation"] for r, c in zip(rotations, cameras)])
    cv_to_camera = np.diag([1., -1., -1.])
    u, _, vt = np.linalg.svd((gt[:, :3, :3] @ cv_to_camera @ rotations).mean(axis=0))
    u[:, -1] *= np.linalg.det(u @ vt)
    rotation = u @ vt
    target = gt[:, :3, 3]
    centered = (centers - centers.mean(axis=0)) @ rotation.T
    variance = np.sum(centered ** 2)
    if variance <= np.finfo(float).eps:
        raise ValueError("accepted cameras have no measurable baseline")
    scale = np.sum(centered * (target - target.mean(axis=0))) / variance
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("camera travel is inconsistent with positive-scale alignment")
    offset = target.mean(axis=0) - scale * rotation @ centers.mean(axis=0)
    # Blender's glTF importer has already mapped (x, -y, -z) to (x, z, -y).
    imported_to_cv = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation @ imported_to_cv
    transform[:3, 3] = offset
    return transform


def score(source, reconstructed, poses, capture, columns):
    width, height = capture["width"], capture["height"]
    rows = max(1, round(columns * height / width))
    tan_x = math.tan(capture["horizontal_fov_radians"] / 2)
    local_rays = [
        ((2 * (x + 0.5) / columns - 1) * tan_x,
         (1 - 2 * (y + 0.5) / rows) * tan_x * height / width, -1)
        for y in range(rows) for x in range(columns)
    ]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    errors = []
    source_hits = extra_hits = 0
    for pose in poses:
        origin = Vector(pose[:3, 3])
        for ray in np.asarray(local_rays) @ pose[:3, :3].T:
            direction = Vector(ray).normalized()
            visible, point, *_ = source.ray_cast(depsgraph, origin, direction)
            distance = reconstructed.ray_cast(origin, direction)[3]
            if not visible:
                extra_hits += int(distance is not None)
                continue
            source_hits += 1
            if distance is not None:
                truth = (point - origin).length
                errors.append(abs(distance - truth) / truth)
    if not source_hits:
        raise ValueError("no reference geometry visible from the capture")
    errors = np.array(errors)
    return {
        "frames": len(poses), "ray_grid": [columns, rows],
        "reference_hits": source_hits, "overlap_hits": len(errors),
        "reconstructed_hits_on_background": extra_hits,
        "visible_ray_coverage": len(errors) / source_hits,
        "median_relative_depth_error_on_overlap": float(np.median(errors)) if len(errors) else None,
        "visible_rays_within_2pct_depth": int((errors <= .02).sum()) / source_hits,
        "visible_rays_within_5pct_depth": int((errors <= .05).sum()) / source_hits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="static .blend or .glb reference")
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--cameras", required=True, type=Path, help="ground-truth sampled camera poses")
    parser.add_argument("--columns", type=int, default=80)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if not 1 <= args.columns <= 640:
        parser.error("--columns must be in 1..640")
    capture = json.loads(args.cameras.read_text())
    poses = np.array([f["matrix_world"] for f in capture["frames"]])
    if (poses.shape != (len(poses), 4, 4) or not np.isfinite(poses).all()
            or min(capture["width"], capture["height"]) <= 0
            or not 0 < capture["horizontal_fov_radians"] < math.pi):
        parser.error("invalid capture dimensions, field of view or camera matrices")
    cameras, vertices, faces = load_export(args.glb.resolve())
    transform = align(cameras, poses)
    world = vertices @ transform[:3, :3].T + transform[:3, 3]
    reconstructed = BVHTree.FromPolygons(world.tolist(), faces, all_triangles=True)
    path = str(args.source.resolve())
    if args.source.suffix.lower() == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path, load_ui=False, use_scripts=False)
    elif args.source.suffix.lower() == ".glb":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        parser.error("expected a .blend or .glb source")
    if "scene" in capture:
        bpy.context.window.scene = bpy.data.scenes[capture["scene"]]
    if "scene_frame" in capture:
        bpy.context.scene.frame_set(capture["scene_frame"])
    report = score(bpy.context.scene, reconstructed, poses, capture, args.columns)
    report["registered_frames"] = [c["frame_index"] for c in cameras]
    report["imported_mesh_to_reference_world"] = transform.tolist()
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
