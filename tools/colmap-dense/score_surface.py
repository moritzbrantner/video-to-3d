# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "pycolmap==4.2.1", "Pillow==12.1.0", "scipy==1.18.1", "trimesh==5.2.1", "embreex==4.4.0.post1",
# ]
# ///
"""Raycast the observed GLB in official TUM camera poses against held-out Kinect depth."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap
from scipy.spatial.transform import Rotation
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from score_tum import timestamped_files


def load_surface(path):
    with path.open("rb") as stream:
        magic, version, _, length, kind = struct.unpack("<4s4I", stream.read(20))
        if (magic, version, kind) != (b"glTF", 2, 0x4E4F534A):
            raise ValueError("expected a GLB with a JSON chunk")
        gltf = json.loads(stream.read(length))
    observed = {
        node["name"] for node in gltf["nodes"] if node.get("name") == "accepted-surface"
        or node.get("extras", {}).get("video_to_3d", {}).get("role") == "observed_object_surface"
    }
    scene = trimesh.load_scene(path, process=False, merge_primitives=True, skip_materials=True)
    geometry = []
    for node in scene.graph.nodes_geometry:
        if node in observed:
            transform, key = scene.graph[node]
            mesh = scene.geometry[key].copy()
            mesh.apply_transform(transform)
            geometry.append(mesh)
    if not geometry:
        raise ValueError("no observed surface mesh in GLB")
    return trimesh.util.concatenate(geometry), gltf["asset"]["extras"]["video_to_3d"]["accepted_camera_evidence"]


def score(glb, dataset, frames, evaluation_frames=None):
    mesh, cameras = load_surface(glb)
    receipt = json.loads(frames.read_text())["frames"]
    rgb = timestamped_files(dataset / "rgb.txt")
    depth = timestamped_files(dataset / "depth.txt")
    rgb_times = np.asarray([float(r[0]) for r in rgb])
    depth_times = np.asarray([float(r[0]) for r in depth])
    trajectory = np.loadtxt(dataset / "groundtruth.txt")
    sample_ordinals = np.asarray([
        round(30 * f.get("presented_time", f.get("timestamp_seconds"))) for f in receipt
    ])
    if np.any((sample_ordinals < 0) | (sample_ordinals >= len(rgb))):
        raise ValueError("timestamps outside the TUM RGB movie")
    pose_indices = [np.argmin(abs(trajectory[:, 0] - rgb_times[sample_ordinals[c["frame_index"]]])) for c in cameras]
    gt = trajectory[pose_indices, 1:4]
    centers = np.asarray([-np.reshape(c["rotation"], (3, 3)).T @ c["translation"] for c in cameras])
    alignment = pycolmap.estimate_sim3d(centers, gt)
    if alignment is None or alignment.scale.item() <= 0:
        raise ValueError("could not determine a positive metric alignment from registered cameras")
    # The GLB exporter maps the solver's CV basis to glTF with a half-turn about X.
    mesh.vertices = alignment * (mesh.vertices * np.array([1.0, -1.0, -1.0]))
    intersector = RayMeshIntersector(mesh)
    # Native TUM Freiburg 3 rectified color/depth intrinsics; use integer source pixel centers.
    y, x = np.mgrid[3:480:8, 3:640:8]
    rays = np.column_stack(((x.ravel() - 320.1) / 535.4, (y.ravel() - 247.6) / 539.2, np.ones(x.size)))
    report = {}
    if evaluation_frames is None:
        evaluation_ordinals = sample_ordinals
    else:
        evaluation = json.loads(evaluation_frames.read_text())["frames"]
        evaluation_ordinals = np.asarray([round(30 * f.get("presented_time", f.get("timestamp_seconds"))) for f in evaluation])
        if np.any((evaluation_ordinals < 0) | (evaluation_ordinals >= len(rgb))):
            raise ValueError("evaluation timestamps outside the TUM RGB movie")
    middle = np.unique((evaluation_ordinals[:-1] + evaluation_ordinals[1:]) // 2)
    for group, ordinals in (("sampled", np.unique(evaluation_ordinals)), ("between_sampled", np.setdiff1d(middle, np.union1d(evaluation_ordinals, sample_ordinals)))):
        errors = []
        valid = 0
        considered = 0
        for ordinal in ordinals:
            time = rgb_times[ordinal]
            index = np.argmin(abs(depth_times - time))
            pose = trajectory[np.argmin(abs(trajectory[:, 0] - time))]
            if abs(depth_times[index] - time) > 0.03 or abs(pose[0] - time) > 0.03:
                continue
            with Image.open(dataset / depth[index][1]) as image:
                distances = np.asarray(image)[y, x].ravel() / 5000.0
            valid += int((distances > 0).sum())
            considered += 1
            orientation = Rotation.from_quat(pose[4:]).as_matrix()
            directions = rays @ orientation.T
            origins = np.broadcast_to(pose[1:4], directions.shape)
            hits, ids, _ = intersector.intersects_location(origins, directions, multiple_hits=False)
            depths = (hits - origins[ids]) @ orientation[:, 2]
            keep = distances[ids] > 0
            errors.append(abs(depths[keep] - distances[ids[keep]]))
        error = np.concatenate(errors) if errors else np.empty(0)
        report[group] = {
            "views": considered, "valid_depth_rays": valid, "surface_hits_with_valid_depth": len(error),
            "coverage": len(error) / valid if valid else None,
            "median_absolute_error_m": float(np.median(error)) if len(error) else None,
            "p90_absolute_error_m": float(np.percentile(error, 90)) if len(error) else None,
            "fraction_of_valid_depth_within_10cm": float((error <= 0.1).sum() / valid) if valid else None,
        }
    return {"surface_triangles": len(mesh.faces), "registered_cameras": len(cameras), "camera_center_rmse_m": float(np.sqrt(np.mean(np.sum((alignment * centers - gt) ** 2, axis=1)))), **report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("glb", "dataset", "frames"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--evaluation-frames", type=Path, help="score the same camera times across different sampling profiles")
    print(json.dumps(score(**vars(parser.parse_args())), indent=2, allow_nan=False))
