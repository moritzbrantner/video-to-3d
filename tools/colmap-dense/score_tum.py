# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["pycolmap==4.2.1", "Pillow==12.1.0"]
# ///
"""Compare COLMAP output to held-out axial Kinect depth on the TUM office RGB movie."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap


def timestamped_files(path):
    return [row.split() for row in path.read_text().splitlines() if row and not row.startswith("#")]


def project_cloud(points, image, camera):
    xyz = image.cam_from_world() * points
    xyz = xyz[xyz[:, 2] > 0]
    pixels = np.floor(camera.img_from_cam(xyz)).astype(int)
    x, y = pixels.T
    inside = (x >= 0) & (y >= 0) & (x < camera.width) & (y < camera.height)
    # Multiple points may land on a single pixel. Compare the closest observed surface.
    z = np.full((camera.height, camera.width), np.inf)
    np.minimum.at(z, (y[inside], x[inside]), xyz[inside, 2])
    return z


def source_depth(path, source_camera, target_camera):
    with Image.open(path) as image:
        raw = np.asarray(image)
    y, x = np.indices((target_camera.height, target_camera.width))
    rays = target_camera.cam_from_img(np.column_stack((x.ravel() + 0.5, y.ravel() + 0.5)))
    xy = source_camera.img_from_cam(np.column_stack((rays, np.ones(len(rays)))))
    xy *= [raw.shape[1] / source_camera.width, raw.shape[0] / source_camera.height]
    x, y = np.floor(xy).astype(int).T
    valid = (x >= 0) & (x < raw.shape[1]) & (y >= 0) & (y < raw.shape[0])
    out = np.zeros(len(x))
    out[valid] = raw[y[valid], x[valid]] / 5000.0
    return out.reshape((target_camera.height, target_camera.width))


def score(workspace, sparse, dataset, frames):
    model = pycolmap.Reconstruction(workspace / "sparse")
    original = pycolmap.Reconstruction(sparse)
    measured = {p.name.removesuffix(".geometric.bin") for p in (workspace / "stereo/depth_maps").glob("*.geometric.bin")}
    images = sorted((im for im in model.images.values() if im.name in measured), key=lambda image: image.name)
    receipt = json.loads(frames.read_text())["frames"]
    rgb = timestamped_files(dataset / "rgb.txt")
    depths = timestamped_files(dataset / "depth.txt")
    depth_times = np.array([float(row[0]) for row in depths])
    poses = np.loadtxt(dataset / "groundtruth.txt")
    rgb_times = []
    for image in images:
        record = receipt[int(Path(image.name).stem)]
        time = record.get("presented_time", record.get("timestamp_seconds"))
        ordinal = round(time * 30)  # The approved RGB movie contains one frame per rgb.txt entry.
        if not 0 <= ordinal < len(rgb):
            raise ValueError(f"video time outside TUM RGB sequence: {time}")
        rgb_times.append(float(rgb[ordinal][0]))
    gt_centers = np.asarray([poses[np.argmin(abs(poses[:, 0] - t)), 1:4] for t in rgb_times])
    centers = np.asarray([image.projection_center() for image in images])
    alignment = pycolmap.estimate_sim3d(centers, gt_centers)
    if alignment is None or alignment.scale <= 0:
        raise ValueError("could not determine metric scale from camera centers")
    cloud = pycolmap.Reconstruction()
    cloud.import_PLY(workspace / "fused.ply")
    points = np.asarray([p.xyz for p in cloud.points3D.values()])
    landmarks = np.asarray([p.xyz for p in model.points3D.values()])
    errors = {"sparse_landmarks": [], "fused_cloud": [], "geometric_depth_map": []}
    depth_deltas = []
    valid_count = 0
    for image, time in zip(images, rgb_times):
        depth_index = int(np.argmin(abs(depth_times - time)))
        delta = abs(depth_times[depth_index] - time)
        if delta > 0.03:
            continue
        depth_deltas.append(delta)
        camera = model.cameras[image.camera_id]
        source_camera = original.cameras[original.images[image.image_id].camera_id]
        truth = source_depth(dataset / depths[depth_index][1], source_camera, camera)
        valid_count += int((truth > 0).sum())
        dense = pycolmap.DepthMap()
        dense.read(workspace / "stereo/depth_maps" / (image.name + ".geometric.bin"))
        for name, predicted in (
            ("geometric_depth_map", dense.to_array()),
            ("fused_cloud", project_cloud(points, image, camera)),
            ("sparse_landmarks", project_cloud(landmarks, image, camera)),
        ):
            usable = (truth > 0) & (predicted > 0) & np.isfinite(predicted)
            errors[name].append(abs(predicted[usable] * alignment.scale - truth[usable]))
    if not valid_count:
        raise ValueError("no valid timestamp-matched depth pixels for this workspace")
    result = {}
    for name, per_image in errors.items():
        error = np.concatenate(per_image)
        result[name] = {
            "compared_pixels": len(error), "coverage": len(error) / valid_count,
            "median_absolute_error_m": float(np.median(error)) if len(error) else None,
            "p90_absolute_error_m": float(np.percentile(error, 90)) if len(error) else None,
            "fraction_of_valid_depth_within_10cm": float((error <= 0.1).sum() / valid_count),
        }
    return {
        "stereo_views": len(images), "scored_views": len(depth_deltas), "sampled_views": len(receipt),
        "ground_truth_valid_pixels_in_registered_views": valid_count,
        "camera_alignment_scale": alignment.scale.item(),
        "camera_center_rmse_m": float(np.sqrt(np.mean(np.sum((alignment * centers - gt_centers) ** 2, axis=1)))),
        "max_rgb_depth_timestamp_difference_s": max(depth_deltas),
        "result": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "sparse", "dataset", "frames"):
        parser.add_argument(name, type=Path)
    print(json.dumps(score(**vars(parser.parse_args())), indent=2))
