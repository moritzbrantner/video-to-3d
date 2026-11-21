# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.2.6", "pycolmap==4.2.1", "open3d==0.19.0", "trimesh==4.6.0",
#   "embreex==4.4.0.post1", "scipy==1.16.3", "Pillow==12.1.0", "networkx==3.4.2",
# ]
# ///
"""Controlled pose/depth ablation of the observed-surface route on the pinned TUM RGB movie.

Four input combinations are formed from {reference, estimated} camera poses and {measured,
estimated} depth at the same 32 reference views the production PatchMatch run used:

    reference poses = TUM motion capture; measured depth = registered Kinect depth;
    estimated poses = the accepted local COLMAP registration; estimated depth = COLMAP
    geometric PatchMatch.

TUM trajectory and Kinect depth are evaluation oracles. They never reach the production path;
the oracle combinations exist only to locate which stage loses geometry.

Each combination is fused by the unchanged Rust `colmap-surface` stage and by an Open3D TSDF
reference. COLMAP's own stereo fusion + Poisson mesh of the estimated combination is scored
alongside. Monocular (estimated) reconstructions are aligned to the motion-capture frame once,
by one similarity transform from camera centres. There is no per-frame scale fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from trimesh.ray.ray_pyembree import RayMeshIntersector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_metrics as metrics  # noqa: E402

TUM_K = (535.4, 539.2, 320.1, 247.6)  # Freiburg 3 rectified RGB (= registered depth) intrinsics.
DEPTH_FACTOR = 5000.0
WIDTH, HEIGHT = 640, 480
# Oracle combinations re-render images into an ideal centred camera, because the Rust stage
# models a centred principal point; this keeps reference-pose runs free of calibration error.
CANONICAL_F = (TUM_K[0] + TUM_K[1]) / 2
COMBOS = (
    ("reference_pose", "measured_depth"),
    ("estimated_pose", "measured_depth"),
    ("reference_pose", "estimated_depth"),
    ("estimated_pose", "estimated_depth"),
)


def name(pose, depth):
    return f"{pose}__{depth}"


# ---------------------------------------------------------------------------------------------
# Dataset access


class Tum:
    def __init__(self, root: Path):
        rows = lambda p: [r.split() for r in p.read_text().splitlines() if r and not r.startswith("#")]
        self.root = root
        self.rgb = rows(root / "rgb.txt")
        self.rgb_times = np.array([float(r[0]) for r in self.rgb])
        self.depth = rows(root / "depth.txt")
        self.depth_times = np.array([float(r[0]) for r in self.depth])
        self.trajectory = np.loadtxt(root / "groundtruth.txt")

    def ordinal(self, frame):
        """rgb.txt row of a sampled movie frame record (or a movie time in seconds)."""
        seconds = frame if isinstance(frame, float) else frame.get("presented_time", frame.get("timestamp_seconds"))
        ordinal = round(30 * seconds)  # The approved RGB movie holds one frame per rgb.txt row.
        if not 0 <= ordinal < len(self.rgb):
            raise ValueError(f"movie time outside the TUM RGB sequence: {seconds}")
        return ordinal

    def pose(self, time):
        """Camera-to-world rotation and centre from the nearest motion-capture sample."""
        row = self.trajectory[np.argmin(abs(self.trajectory[:, 0] - time))]
        if abs(row[0] - time) > 0.03:
            return None
        return Rotation.from_quat(row[4:]).as_matrix(), row[1:4].copy()

    def kinect(self, time):
        index = int(np.argmin(abs(self.depth_times - time)))
        if abs(self.depth_times[index] - time) > 0.03:
            return None
        with Image.open(self.root / self.depth[index][1]) as image:
            return np.asarray(image, dtype=np.float64) / DEPTH_FACTOR

    def color(self, ordinal):
        with Image.open(self.root / self.rgb[ordinal][1]) as image:
            return np.asarray(image.convert("RGB"))


def umeyama(source, target):
    """Similarity (s, R, t) minimising |s R source + t - target|; one global alignment."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(b.T @ a / len(source))
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / a.var(0).sum()
    return scale, rotation, mu_t - scale * rotation @ mu_s


# ---------------------------------------------------------------------------------------------
# Input preparation


def sample_tum_pixels(rays_x, rays_y):
    """Continuous TUM pixel coordinates (integer-centre convention) of camera rays."""
    return TUM_K[0] * rays_x + TUM_K[2], TUM_K[1] * rays_y + TUM_K[3]


def canonical_rays(x, y):
    return (x + 0.5 - WIDTH / 2) / CANONICAL_F, (y + 0.5 - HEIGHT / 2) / CANONICAL_F


def bilinear(image, u, v):
    u = np.clip(u, 0, image.shape[1] - 1.001)
    v = np.clip(v, 0, image.shape[0] - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[..., None], (v - y0)[..., None]
    p = lambda yy, xx: image[yy, xx].astype(np.float64)
    top = p(y0, x0) * (1 - fx) + p(y0, x0 + 1) * fx
    bottom = p(y0 + 1, x0) * (1 - fx) + p(y0 + 1, x0 + 1) * fx
    return top * (1 - fy) + bottom * fy


def nearest(image, u, v):
    """Nearest-pixel lookup that never blends across a depth edge; outside -> 0."""
    x, y = np.rint(u).astype(int), np.rint(v).astype(int)
    out = np.zeros(np.shape(u))
    ok = (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
    out[ok] = image[y[ok], x[ok]]
    return out


def write_rgba(path, rgb):
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
    data = np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()
    path.write_bytes(data)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def gauge(cameras_w2c, seed):
    """Express world->camera poses in the Rust unit-baseline gauge of `seed` = (first, second)."""
    (r0, t0), (r1, t1) = cameras_w2c[seed[0]], cameras_w2c[seed[1]]
    c0, c1 = -r0.T @ t0, -r1.T @ t1
    baseline = np.linalg.norm(c1 - c0)
    out = {}
    for frame, (r, t) in cameras_w2c.items():
        out[frame] = (r @ r0.T, (r @ c0 + t) / baseline)
    return out, baseline


def synthesize_landmarks(frames, poses, depth, rng, per_frame=60):
    """Oracle-only sparse tracks consistent with the reference poses, so the unchanged Rust
    registration checks can accept motion-capture cameras. Built from Kinect depth."""
    landmarks = []
    frame_ids = sorted(poses)
    for frame in frame_ids:
        d = depth.get(frame)
        if d is None:
            continue
        ys, xs = np.nonzero(d > 0.3)
        if not len(xs):
            continue
        pick = rng.choice(len(xs), size=min(per_frame, len(xs)), replace=False)
        r, t = poses[frame]
        for i in pick:
            x, y = xs[i], ys[i]
            rx, ry = canonical_rays(x, y)
            world = r.T @ (np.array([rx, ry, 1.0]) * d[y, x] - t)
            observations = []
            for other in frame_ids:
                # Frames without a time-matched Kinect image keep exact projections without an
                # occlusion check; the tracks only let the oracle poses pass registration.
                od = depth.get(other)
                ro, to = poses[other]
                q = ro @ world + to
                if q[2] <= 0.1:
                    continue
                u = CANONICAL_F * q[0] / q[2] + WIDTH / 2
                v = CANONICAL_F * q[1] / q[2] + HEIGHT / 2
                if not (1 <= u < WIDTH - 1 and 1 <= v < HEIGHT - 1):
                    continue
                measured = od[int(v), int(u)] if od is not None else q[2]
                if measured > 0 and abs(measured - q[2]) <= 0.02 * q[2]:
                    observations.append({"frame_index": other, "xy": [u, v]})
            if len(observations) >= 2:
                landmarks.append({"xyz": world.tolist(), "rgb": [128, 128, 128], "observations": observations})
    return landmarks


def prepare(args):
    out = args.out / "inputs"
    tum = Tum(args.dataset)
    manifest = json.loads(args.keyframes.read_text())
    frames = manifest["frames"]
    registration = json.loads(args.registered.read_text())
    grids = json.loads(args.grids.read_text())
    stride, reference_stride = grids["stride"], grids["reference_stride"]
    cols, rows = -(-WIDTH // stride), -(-HEIGHT // stride)
    gy, gx = np.divmod(np.arange(cols * rows), cols)
    gx, gy = gx * stride, gy * stride
    focal_est = registration["focal_pixels"]

    times = {f["frame_index"]: tum.rgb_times[tum.ordinal(f)] for f in frames}
    reference_poses = {}
    for frame, time in times.items():
        pose = tum.pose(time)
        if pose is None:
            raise ValueError(f"no motion-capture pose for frame {frame}")
        r_c2w, centre = pose
        reference_poses[frame] = (r_c2w.T, -r_c2w.T @ centre)

    # One global similarity between estimated gauge centres and motion-capture centres.
    estimated = {c["frame_index"]: (np.reshape(c["rotation"], (3, 3)), np.array(c["translation"])) for c in registration["cameras"]}
    order = sorted(estimated)
    est_centres = np.array([-estimated[f][0].T @ estimated[f][1] for f in order])
    ref_centres = np.array([-reference_poses[f][0].T @ reference_poses[f][1] for f in order])
    scale, rotation, translation = umeyama(est_centres, ref_centres)
    residual = np.linalg.norm(scale * est_centres @ rotation.T + translation - ref_centres, axis=1)
    alignment = {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
                 "centre_rmse_m": float(np.sqrt((residual ** 2).mean())), "cameras": len(order)}

    # The production PatchMatch grids, already in the estimated unit-baseline gauge.
    patch_grid = {m["frame"]: np.array(m["depth"]) for m in grids["maps"]}
    model = pycolmap.Reconstruction(args.sparse)
    images = {int(Path(i.name).stem): i for i in model.images.values()}
    seed = (registration["cameras"][0]["frame_index"], registration["cameras"][1]["frame_index"])
    colmap_baseline = np.linalg.norm(images[seed[1]].projection_center() - images[seed[0]].projection_center())
    dense_model = pycolmap.Reconstruction(args.stereo / "sparse")
    dense_size = next(iter(dense_model.cameras.values())).width / WIDTH

    def patch_full(frame):
        """Raw PatchMatch depth (estimated gauge) for a continuous image pixel grid."""
        depth = pycolmap.DepthMap()
        depth.read(args.stereo / "stereo/depth_maps" / f"{frame:08d}.png.geometric.bin")
        return depth.to_array() / colmap_baseline

    keyframe_root = args.keyframes.parent
    kinect = {f: tum.kinect(times[f]) for f in times}
    reference_frames = sorted(patch_grid)
    summary = {"alignment": alignment, "reference_frames": reference_frames, "stride": stride,
               "reference_stride": reference_stride, "combos": {}}

    canonical_depth = {}
    rng = np.random.default_rng(7)
    for pose_kind in ("reference_pose", "estimated_pose"):
        canonical = pose_kind == "reference_pose"
        # Frames: canonical warp for reference poses, untouched production pixels otherwise.
        frame_dir = out / pose_kind / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for f in frames:
            target = frame_dir / f"{f['frame_index']}.rgba"
            if canonical:
                rgba = np.frombuffer((keyframe_root / f["pixels"]).read_bytes(), np.uint8).reshape(HEIGHT, WIDTH, 4)
                yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
                u, v = sample_tum_pixels(*canonical_rays(xx, yy))
                content_hash = write_rgba(target, np.clip(np.rint(bilinear(rgba[..., :3], u, v)), 0, 255).astype(np.uint8))
                if kinect[f["frame_index"]] is not None:
                    canonical_depth[f["frame_index"]] = nearest(kinect[f["frame_index"]], u, v)
            else:
                if not target.exists():
                    target.symlink_to((keyframe_root / f["pixels"]).resolve())
                content_hash = f["content_hash"]
            records.append({**f, "pixels": target.name, "content_hash": content_hash})
        (frame_dir / "keyframes.json").write_text(json.dumps({**manifest, "frames": records}))

        if canonical:
            gauge_poses, baseline = gauge(reference_poses, seed)
            landmarks = synthesize_landmarks(frames, gauge_poses, {k: v / baseline for k, v in canonical_depth.items()}, rng)
            cams = [seed[0], seed[1], *[f for f in sorted(gauge_poses) if f not in seed]]
            reg = {
                "schema_version": 1, "provider": registration["provider"], "version": registration["version"],
                "frames": [{"frame_index": r["frame_index"], "content_hash": r["content_hash"]} for r in records],
                "width": WIDTH, "height": HEIGHT, "focal_pixels": CANONICAL_F,
                "cameras": [{"frame_index": f, "rotation": gauge_poses[f][0].ravel().tolist(),
                             "translation": gauge_poses[f][1].tolist()} for f in cams],
                "landmarks": landmarks,
            }
            to_metric = baseline
        else:
            gauge_poses = estimated
            reg = registration
            to_metric = scale
        (out / pose_kind / "registered.json").write_text(json.dumps(reg))

        # World->camera extrinsics in the motion-capture metric frame for TSDF fusion.
        if canonical:
            metric_w2c = reference_poses
        else:
            metric_w2c = {f: (r @ rotation.T, -r @ rotation.T @ translation + scale * t) for f, (r, t) in estimated.items()}

        for depth_kind in ("measured_depth", "estimated_depth"):
            combo = out / name(pose_kind, depth_kind)
            combo.mkdir(parents=True, exist_ok=True)
            maps, views = [], {}
            for m in grids["maps"]:
                frame = m["frame"]
                if canonical:
                    rx, ry = canonical_rays(gx, gy)
                    u, v = sample_tum_pixels(rx, ry)
                    if depth_kind == "measured_depth":
                        k = kinect[frame]
                        metric = nearest(k, u, v) if k is not None else np.zeros(len(gx))
                    else:
                        raw = patch_full(frame)
                        # Same physical image location, then the one global scale.
                        du = np.floor((u + 0.5) * dense_size).astype(int)
                        dv = np.floor((v + 0.5) * dense_size).astype(int)
                        ok = (du >= 0) & (dv >= 0) & (du < raw.shape[1]) & (dv < raw.shape[0])
                        metric = np.zeros(len(gx))
                        metric[ok] = raw[dv[ok], du[ok]] * scale
                    focal = CANONICAL_F
                else:
                    if depth_kind == "measured_depth":
                        k = kinect[frame]
                        metric = k[gy, gx] if k is not None else np.zeros(len(gx))
                    else:
                        metric = patch_grid[frame] * scale
                    focal = focal_est
                metric = np.where(np.isfinite(metric) & (metric > 0), metric, 0.0)
                maps.append({"frame": frame, "sources": m["sources"], "depth": (metric / to_metric).round(6).tolist()})
                views[str(frame)] = metric.reshape(rows, cols)
            grid_doc = {"version": 1, "stride": stride, "reference_stride": reference_stride, "maps": maps}
            (combo / "grids.json").write_text(json.dumps(grid_doc, separators=(",", ":")))
            np.savez_compressed(combo / "views.npz", focal=focal, **views,
                                **{f"pose_{f}": np.hstack([metric_w2c[int(f)][0], metric_w2c[int(f)][1][:, None]]) for f in views})
            summary["combos"][name(pose_kind, depth_kind)] = {
                "frames": str(frame_dir / "keyframes.json"), "registered": str(out / pose_kind / "registered.json"),
                "grids": str(combo / "grids.json"), "valid_reference_samples": int(sum((v > 0).sum() for v in views.values())),
            }
        if canonical:
            summary["oracle_landmarks"] = len(landmarks)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "combos"}, indent=2))


def prepare_learned(args):
    """Add MapAnything depth at the estimated poses, scaled once without any oracle.

    The single scale is the median ratio between accepted COLMAP landmark depths and learned
    depth at their observations in the reference views. Per-group ratios are reported, not used.
    `learned_depth` uses learned depth alone; `filled_depth` keeps PatchMatch and uses learned
    depth only where PatchMatch has no measurement (Atlas 2.8.1: escalate only for missing
    evidence). Both still go through the unchanged Rust cross-view verification.
    """
    out = args.out / "inputs"
    summary = json.loads((out / "summary.json").read_text())
    registration = json.loads(args.registered.read_text())
    base = out / "estimated_pose__estimated_depth"
    production = np.load(base / "views.npz")
    grids = json.loads((base / "grids.json").read_text())
    report = json.loads((args.out / "mapanything/report.json").read_text())
    model_w, model_h = report["model_size"]
    learned = {f: np.load(args.out / "mapanything" / f"{f}.npz")["depth"] for f in summary["reference_frames"]}
    poses = {c["frame_index"]: (np.reshape(c["rotation"], (3, 3)), np.array(c["translation"])) for c in registration["cameras"]}

    def lookup(frame, x, y):
        u = np.rint((np.asarray(x) + 0.5) * model_w / WIDTH - 0.5).astype(int)
        v = np.rint((np.asarray(y) + 0.5) * model_h / HEIGHT - 0.5).astype(int)
        return learned[frame][np.clip(v, 0, model_h - 1), np.clip(u, 0, model_w - 1)]

    ratios = {f: [] for f in learned}
    for landmark in registration["landmarks"]:
        for ob in landmark["observations"]:
            frame = ob["frame_index"]
            if frame not in learned:
                continue
            r, t = poses[frame]
            z = (r @ np.array(landmark["xyz"]) + t)[2]
            d = lookup(frame, ob["xy"][0] - 0.5, ob["xy"][1] - 0.5)
            if z > 0 and d > 0:
                ratios[frame].append(z / d)
    every = np.concatenate([np.array(v) for v in ratios.values() if v])
    scale_to_gauge = float(np.median(every))
    metric_scale = summary["alignment"]["scale"]
    stride, reference_stride = grids["stride"], grids["reference_stride"]
    cols = -(-WIDTH // stride)
    groups = [summary["reference_frames"][i:i + report["group"]] for i in range(0, len(summary["reference_frames"]), report["group"])]
    learned_scales = {
        "landmark_observations": int(len(every)), "global_scale_to_gauge": scale_to_gauge,
        "per_group_median_ratio_relative_to_global": [
            float(np.median(np.concatenate([ratios[f] for f in g if ratios[f]])) / scale_to_gauge) for g in groups],
    }
    for kind in ("learned_depth", "filled_depth"):
        combo = out / f"estimated_pose__{kind}"
        combo.mkdir(parents=True, exist_ok=True)
        maps, views = [], {}
        for m in grids["maps"]:
            frame = m["frame"]
            i = np.arange(len(m["depth"]))
            gx, gy = (i % cols) * stride, (i // cols) * stride
            depth = lookup(frame, gx, gy) * scale_to_gauge
            if kind == "filled_depth":
                patch = np.array(m["depth"])
                depth = np.where(patch > 0, patch, depth)
            depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
            maps.append({"frame": frame, "sources": m["sources"], "depth": depth.round(6).tolist()})
            views[str(frame)] = (depth * metric_scale).reshape(production[str(frame)].shape)
        (combo / "grids.json").write_text(json.dumps({**grids, "maps": maps}, separators=(",", ":")))
        np.savez_compressed(combo / "views.npz", focal=production["focal"], **views,
                            **{k: production[k] for k in production.files if k.startswith("pose_")})
        summary["combos"][f"estimated_pose__{kind}"] = {
            **summary["combos"]["estimated_pose__estimated_depth"], "grids": str(combo / "grids.json"),
            "valid_reference_samples": int(sum((v > 0).sum() for v in views.values())),
        }
    summary["learned_scale"] = learned_scales
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(learned_scales, indent=2))


# ---------------------------------------------------------------------------------------------
# Fusion


def run_rust(args):
    summary = json.loads((args.out / "inputs/summary.json").read_text())
    for combo, paths in summary["combos"].items():
        target = args.out / "surfaces" / f"rust__{combo}.glb"
        report = target.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        result = subprocess.run([args.cli, "colmap-surface", paths["frames"], paths["registered"], paths["grids"], str(target)],
                                capture_output=True, text=True)
        report.write_text(result.stdout)
        print(combo, "exit", result.returncode, result.stdout[:200] if result.returncode else "")


def tsdf(args):
    summary = json.loads((args.out / "inputs/summary.json").read_text())
    stride = summary["stride"]
    for combo in summary["combos"]:
        views = np.load(args.out / "inputs" / combo / "views.npz")
        focal = float(views["focal"])
        frames_doc = json.loads(Path(summary["combos"][combo]["frames"]).read_text())
        root = Path(summary["combos"][combo]["frames"]).parent
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=args.voxel, sdf_trunc=4 * args.voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for key in summary["reference_frames"]:
            depth = views[str(key)].astype(np.float32)
            rows, cols = depth.shape
            record = next(f for f in frames_doc["frames"] if f["frame_index"] == key)
            rgba = np.frombuffer((root / record["pixels"]).read_bytes(), np.uint8).reshape(HEIGHT, WIDTH, 4)
            color = np.ascontiguousarray(rgba[::stride, ::stride, :3])
            intrinsic = o3d.camera.PinholeCameraIntrinsic(cols, rows, focal / stride, focal / stride,
                                                          (WIDTH / 2 - 0.5) / stride, (HEIGHT / 2 - 0.5) / stride)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color), o3d.geometry.Image(depth), depth_scale=1.0, depth_trunc=6.0,
                convert_rgb_to_intensity=False)
            extrinsic = np.eye(4)
            extrinsic[:3] = views[f"pose_{key}"]
            volume.integrate(rgbd, intrinsic, extrinsic)
        mesh = volume.extract_triangle_mesh()
        target = args.out / "surfaces" / f"tsdf__{combo}.ply"
        target.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(target), mesh)
        print(combo, len(mesh.triangles), "triangles")


def reference_surface(args):
    """Dense Kinect TSDF over the whole clip at motion-capture poses (precision oracle only)."""
    tum = Tum(args.dataset)
    frames = json.loads(args.keyframes.read_text())["frames"]
    start = tum.rgb_times[tum.ordinal(frames[0])]
    end = tum.rgb_times[tum.ordinal(frames[-1])]
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.04, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, *TUM_K)
    used = 0
    for index in range(0, len(tum.depth), 6):
        time = tum.depth_times[index]
        if not start <= time <= end:
            continue
        pose = tum.pose(time)
        ordinal = int(np.argmin(abs(tum.rgb_times - time)))
        if pose is None or abs(tum.rgb_times[ordinal] - time) > 0.03:
            continue
        depth = tum.kinect(time).astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(tum.color(ordinal))), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False)
        r, c = pose
        extrinsic = np.eye(4)
        extrinsic[:3, :3], extrinsic[:3, 3] = r.T, -r.T @ c
        volume.integrate(rgbd, intrinsic, extrinsic)
        used += 1
    mesh = volume.extract_triangle_mesh()
    target = args.out / "surfaces/reference_kinect_full.ply"
    target.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(target), mesh)
    print("reference frames", used, "triangles", len(mesh.triangles))


def colmap_reference(args):
    """COLMAP's own fusion and Poisson mesh of the same PatchMatch workspace, metric-aligned once."""
    tum = Tum(args.dataset)
    frames = {f["frame_index"]: f for f in json.loads(args.keyframes.read_text())["frames"]}
    model = pycolmap.Reconstruction(args.sparse)
    images = sorted(model.images.values(), key=lambda i: i.name)
    centres, truth = [], []
    for image in images:
        frame = frames[int(Path(image.name).stem)]
        pose = tum.pose(tum.rgb_times[tum.ordinal(frame)])
        centres.append(image.projection_center())
        truth.append(pose[1])
    scale, rotation, translation = umeyama(np.array(centres), np.array(truth))
    poisson = args.out / "surfaces/colmap_poisson_raw.ply"
    # COLMAP's defaults (octree depth 13, trim 10) return an empty mesh for these ~33k fused
    # points at 360 px; depth 9 / trim 7 is the least extrapolating setting that yields a surface.
    options = pycolmap.PoissonMeshingOptions()
    options.depth, options.trim = 9, 7.0
    pycolmap.poisson_meshing(args.stereo / "fused.ply", poisson, options)
    for source, target in ((poisson, "colmap_poisson__estimated_pose__estimated_depth.ply"),
                           (args.stereo / "fused.ply", "colmap_fused_points__estimated_pose__estimated_depth.ply")):
        if "fused" in target:
            geometry = o3d.io.read_point_cloud(str(source))
        else:
            geometry = o3d.io.read_triangle_mesh(str(source))
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = scale * rotation, translation
        geometry.transform(transform)
        writer = o3d.io.write_point_cloud if "fused" in target else o3d.io.write_triangle_mesh
        writer(str(args.out / "surfaces" / target), geometry)
    print("colmap alignment scale", scale)


# ---------------------------------------------------------------------------------------------
# Evaluation


class Surface:
    """A triangle surface in the metric motion-capture frame, with optional colour lookup."""

    def __init__(self, mesh, colorize, double_sided=True):
        self.mesh = mesh
        self.colorize = colorize  # (face_ids, locations) -> rgb
        self.double_sided = double_sided
        self.intersector = RayMeshIntersector(mesh)


def load_glb(path, tum, frames):
    with path.open("rb") as stream:
        magic, version, _, length, _ = struct.unpack("<4s4I", stream.read(20))
        gltf = json.loads(stream.read(length))
    observed = {n["name"] for n in gltf["nodes"] if n.get("name") == "accepted-surface"
                or n.get("extras", {}).get("video_to_3d", {}).get("role") == "observed_object_surface"}
    double_sided = all(m.get("doubleSided", False) for m in gltf.get("materials", []))
    cameras = gltf["asset"]["extras"]["video_to_3d"]["accepted_camera_evidence"]
    scene = trimesh.load_scene(path, process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        # Multi-primitive nodes (one primitive per texture page) load with a hash suffix.
        if node not in observed and node.rsplit("_", 1)[0] not in observed:
            continue
        transform, key = scene.graph[node]
        part = scene.geometry[key].copy()
        part.apply_transform(transform)
        parts.append(part)
    centres = np.array([-np.reshape(c["rotation"], (3, 3)).T @ c["translation"] for c in cameras])
    by_frame = {f["frame_index"]: f for f in frames}
    truth = np.array([tum.pose(tum.rgb_times[tum.ordinal(by_frame[c["frame_index"]])])[1] for c in cameras])
    scale, rotation, translation = umeyama(centres, truth)
    flip = np.array([1.0, -1.0, -1.0])  # The GLB exporter turns the solver's CV basis half about X.
    offsets, textures = [], []
    for part in parts:
        part.vertices = scale * (part.vertices * flip) @ rotation.T + translation
        uv = getattr(part.visual, "uv", None)
        material = getattr(part.visual, "material", None)
        image = getattr(material, "baseColorTexture", None) if material is not None else None
        textures.append((uv, np.asarray(image.convert("RGB")) if image is not None else None))
    mesh = trimesh.util.concatenate([trimesh.Trimesh(p.vertices, p.faces, process=False) for p in parts])
    face_part = np.concatenate([np.full(len(p.faces), i) for i, p in enumerate(parts)])
    face_offset = np.concatenate([[0], np.cumsum([len(p.faces) for p in parts])])

    def colorize(face_ids, locations):
        out = np.zeros((len(face_ids), 3))
        for i, part in enumerate(parts):
            sel = face_part[face_ids] == i
            uv, image = textures[i]
            if not sel.any() or uv is None or image is None:
                continue
            local = face_ids[sel] - face_offset[i]
            tri = part.faces[local]
            bary = trimesh.triangles.points_to_barycentric(part.vertices[tri], locations[sel])
            coords = np.einsum("ij,ijk->ik", bary, uv[tri])
            u = coords[:, 0] * (image.shape[1] - 1)
            v = (1 - coords[:, 1]) * (image.shape[0] - 1)
            out[sel] = bilinear(image, u, v)
        return out

    return Surface(mesh, colorize, double_sided), {"glb_double_sided": double_sided, "camera_centre_rmse_m": float(
        np.sqrt((np.linalg.norm(scale * centres @ rotation.T + translation - truth, axis=1) ** 2).mean()))}


def load_ply(path):
    raw = o3d.io.read_triangle_mesh(str(path))
    mesh = trimesh.Trimesh(np.asarray(raw.vertices), np.asarray(raw.triangles), process=False)
    colors = np.asarray(raw.vertex_colors) * 255 if raw.has_vertex_colors() else None

    def colorize(face_ids, locations):
        if colors is None:
            return np.full((len(face_ids), 3), 160.0)
        tri = mesh.faces[face_ids]
        bary = trimesh.triangles.points_to_barycentric(mesh.vertices[tri], locations)
        return np.einsum("ij,ijk->ik", bary, colors[tri])

    return Surface(mesh, colorize), {}


def cast(surface, centre, r_c2w, rays):
    directions = rays @ r_c2w.T
    origins = np.broadcast_to(centre, directions.shape)
    locations, ray_ids, face_ids = surface.intersector.intersects_location(origins, directions, multiple_hits=False)
    depth = np.full(len(rays), np.nan)
    depth[ray_ids] = (locations - centre) @ r_c2w[:, 2]
    facing = np.zeros(len(rays), bool)
    facing[ray_ids] = np.einsum("ij,ij->i", surface.mesh.face_normals[face_ids], directions[ray_ids]) < 0
    return depth, facing, ray_ids, face_ids, locations


def tum_rays(step, offset):
    y, x = np.mgrid[offset:HEIGHT:step, offset:WIDTH:step]
    rays = np.column_stack(((x.ravel() - TUM_K[2]) / TUM_K[0], (y.ravel() - TUM_K[3]) / TUM_K[1], np.ones(x.size)))
    return rays, y.ravel(), x.ravel(), y.shape


def evaluation_ordinals(tum, keyframes, evaluation):
    sampled = np.unique([tum.ordinal(f) for f in evaluation])
    keys = np.array([tum.ordinal(f) for f in keyframes])
    middle = np.unique((sampled[:-1] + sampled[1:]) // 2)
    return sampled, np.setdiff1d(middle, np.union1d(sampled, keys))


def topology(mesh):
    merged = mesh.copy()
    merged.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=5)
    components = trimesh.graph.connected_components(merged.face_adjacency, nodes=np.arange(len(merged.faces)), min_len=1)
    sizes = np.sort([len(c) for c in components])[::-1]
    edges = merged.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    lengths = np.linalg.norm(merged.vertices[boundary[:, 0]] - merged.vertices[boundary[:, 1]], axis=1)
    loops = 0
    if len(boundary):
        graph = trimesh.graph.nx.Graph()
        graph.add_edges_from(boundary.tolist())
        loops = trimesh.graph.nx.number_connected_components(graph)
    return {
        "triangles": len(mesh.faces), "area_m2": float(mesh.area),
        "components": len(sizes), "components_under_32_triangles": int((sizes < 32).sum()),
        "largest_component_share": float(sizes[0] / sizes.sum()) if len(sizes) else 0.0,
        "share_in_components_under_32": float(sizes[sizes < 32].sum() / sizes.sum()) if len(sizes) else 0.0,
        "boundary_loops": int(loops), "boundary_length_m": float(lengths.sum()),
        "boundary_length_per_m2": float(lengths.sum() / mesh.area) if mesh.area else None,
    }


class Reference:
    def __init__(self, path, tum, ordinals):
        raw = o3d.io.read_triangle_mesh(str(path))
        mesh = trimesh.Trimesh(np.asarray(raw.vertices), np.asarray(raw.triangles), process=False)
        self.points, _ = trimesh.sample.sample_surface(mesh, 1_500_000, seed=3)
        self.tree = cKDTree(self.points)
        # Free-space witnesses: motion-capture Kinect views spread over the clip.
        self.witnesses = []
        for ordinal in ordinals:
            time = tum.rgb_times[ordinal]
            pose, depth = tum.pose(time), tum.kinect(time)
            if pose is not None and depth is not None:
                self.witnesses.append((pose, depth))
        self.planes = self._planes()

    def _planes(self):
        o3d.utility.random.seed(5)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points)).voxel_down_sample(0.02)
        remaining = cloud
        planes = []
        total = len(cloud.points)
        for _ in range(16):
            if len(planes) == 5 or len(remaining.points) < 0.015 * total:
                break
            model, inliers = remaining.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=2000)
            normal = np.asarray(model[:3]) / np.linalg.norm(model[:3])
            offset = model[3] / np.linalg.norm(model[:3])
            if normal[2] < 0:
                normal, offset = -normal, -offset
            # Remove the whole slab, so an uneven floor is not found again as several planes.
            slab = np.nonzero(np.abs(np.asarray(remaining.points) @ normal + offset) < 0.06)[0]
            inlier_cloud = remaining.select_by_index(inliers)
            remaining = remaining.select_by_index(slab, invert=True)
            if len(inliers) < 0.015 * total:
                continue
            # Keep the largest spatially connected piece, so two parallel desks are not merged.
            labels = np.asarray(inlier_cloud.cluster_dbscan(eps=0.05, min_points=5))
            if labels.max() < 0:
                continue
            keep = labels == np.bincount(labels[labels >= 0]).argmax()
            points = np.asarray(inlier_cloud.points)[keep]
            if len(points) < 0.015 * total:
                continue
            # The motion-capture frame is z-up with the floor near z = 0.
            if abs(normal[2]) > 0.9:
                height = float(points[:, 2].mean())
                label = "floor" if height < 0.2 else f"horizontal surface at {height:.2f} m"
            else:
                label = "vertical surface" if abs(normal[2]) < 0.2 else "inclined surface"
            planes.append({"label": label, "normal": normal, "offset": offset, "points": points})
        return planes

    def precision(self, mesh, rng):
        samples, _ = trimesh.sample.sample_surface(mesh, 200_000, seed=int(rng.integers(1 << 30)))
        distance, _ = self.tree.query(samples, workers=-1)
        seen = np.zeros(len(samples), int)
        contradicted = np.zeros(len(samples), int)
        for (r, c), depth in self.witnesses:
            q = (samples - c) @ r
            ok = q[:, 2] > 0.2
            u = np.rint(TUM_K[0] * q[:, 0] / np.where(ok, q[:, 2], 1) + TUM_K[2]).astype(int)
            v = np.rint(TUM_K[1] * q[:, 1] / np.where(ok, q[:, 2], 1) + TUM_K[3]).astype(int)
            ok &= (u >= 0) & (v >= 0) & (u < WIDTH) & (v < HEIGHT)
            measured = np.zeros(len(samples))
            measured[ok] = depth[v[ok], u[ok]]
            ok &= measured > 0
            seen += ok
            contradicted += ok & (q[:, 2] < measured - metrics.TOLERANCE_M)
        free_space = (seen >= 2) & (contradicted * 2 > seen)
        return {
            "precision_within_5cm": float((distance <= metrics.TIGHT_TOLERANCE_M).mean()),
            "precision_within_10cm": float((distance <= metrics.TOLERANCE_M).mean()),
            "free_space_violation": float(free_space.mean()),
            "unverifiable": float(((distance > metrics.TOLERANCE_M) & ~free_space).mean()),
            "median_distance_to_reference_m": float(np.median(distance)),
        }, samples, distance

    def plane_errors(self, samples):
        tree = cKDTree(samples)
        out = []
        for i, plane in enumerate(self.planes):
            # Coverage: reference plane points with a reconstructed sample within 3 cm.
            distance, _ = tree.query(plane["points"], workers=-1)
            footprint = cKDTree(plane["points"])
            near, _ = footprint.query(samples, distance_upper_bound=0.05, workers=-1)
            attributed = samples[np.isfinite(near)]
            offsets = np.abs(attributed @ plane["normal"] + plane["offset"])
            out.append({
                "plane": i, "label": plane["label"], "normal": np.round(plane["normal"], 3).tolist(), "reference_points": len(plane["points"]),
                "coverage_within_3cm": float((distance <= 0.03).mean()),
                "median_plane_error_m": float(np.median(offsets)) if len(offsets) else None,
                "p90_plane_error_m": float(np.percentile(offsets, 90)) if len(offsets) else None,
                "samples_off_plane_over_5cm": float((offsets > 0.05).mean()) if len(offsets) else None,
            })
        return out


def evaluate(args):
    tum = Tum(args.dataset)
    keyframes = json.loads(args.keyframes.read_text())["frames"]
    evaluation = json.loads(args.evaluation.read_text())["frames"]
    sampled, held_out = evaluation_ordinals(tum, keyframes, evaluation)
    rays, ys, xs, _ = tum_rays(8, 3)
    surfaces = sorted((args.out / "surfaces").glob("*__*.*"))
    surfaces = [p for p in surfaces if p.suffix in (".glb", ".ply") and "fused_points" not in p.name and "raw" not in p.name]
    if args.only:
        surfaces = [p for p in surfaces if any(o in p.name for o in args.only)]
    reference = Reference(args.out / "surfaces/reference_kinect_full.ply", tum, held_out[::2])
    results_path = args.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    rng = np.random.default_rng(11)
    truths = {}
    for ordinal in np.union1d(sampled, held_out):
        time = tum.rgb_times[ordinal]
        pose, depth = tum.pose(time), tum.kinect(time)
        if pose is not None and depth is not None:
            truths[int(ordinal)] = (pose, depth[ys, xs], tum.color(ordinal)[ys, xs].astype(float))
    for path in surfaces:
        label = path.stem
        if path.suffix == ".glb":
            surface, extra = load_glb(path, tum, keyframes)
        else:
            surface, extra = load_ply(path)
        entry = {"surface": str(path.relative_to(args.out)), **extra}
        for group, ordinals in (("sampled", sampled), ("held_out", held_out)):
            counts = dict.fromkeys(metrics.RAY_CLASSES, 0)
            errors, front_only, photometric, valid_total = [], 0, [], 0
            tight = 0
            for ordinal in ordinals:
                if int(ordinal) not in truths:
                    continue
                (r, c), truth, color = truths[int(ordinal)]
                depth, facing, ray_ids, face_ids, locations = cast(surface, c, r, rays)
                valid = truth > 0
                valid_total += int(valid.sum())
                hit = np.isfinite(depth)
                error = depth - truth
                counts["missing"] += int((valid & ~hit).sum())
                good = valid & hit & (np.abs(error) <= metrics.TOLERANCE_M)
                counts["correct"] += int(good.sum())
                counts["false_positive_in_front"] += int((valid & hit & (error < -metrics.TOLERANCE_M)).sum())
                counts["behind_truth"] += int((valid & hit & (error > metrics.TOLERANCE_M)).sum())
                tight += int((valid & hit & (np.abs(error) <= metrics.TIGHT_TOLERANCE_M)).sum())
                front_only += int((good & facing).sum())
                errors.append(np.abs(error[valid & hit]))
                if group == "held_out":
                    rendered = np.zeros((len(rays), 3))
                    rendered[ray_ids] = surface.colorize(face_ids, locations)
                    photometric.append(np.abs(rendered[good] - color[good]).mean(1))
            error = np.concatenate(errors) if errors else np.empty(0)
            summary = metrics.ray_summary(counts)
            entry[group] = {
                "views": int(sum(int(o) in truths for o in ordinals)), "valid_rays": valid_total,
                "coverage": 1 - summary["missing"], **summary,
                "correct_within_5cm": tight / valid_total,
                "median_abs_depth_error_m": float(np.median(error)) if len(error) else None,
                "p90_abs_depth_error_m": float(np.percentile(error, 90)) if len(error) else None,
                "correct_visible_with_backface_culling": front_only / valid_total,
            }
            if photometric:
                p = np.concatenate(photometric)
                entry[group]["held_out_rgb_mae"] = float(p.mean())
        entry["topology"] = topology(surface.mesh)
        entry["precision"], samples, _ = reference.precision(surface.mesh, rng)
        entry["planes"] = reference.plane_errors(samples)
        results[label] = entry
        print(label, json.dumps({k: entry["held_out"][k] for k in ("coverage", "correct", "false_positive_in_front", "behind_truth")}))
        results_path.write_text(json.dumps(results, indent=2))
    # The depth maps themselves, before any fusion, for the stage attribution.
    results["_planes"] = [{"label": p["label"], "normal": np.round(p["normal"], 3).tolist(), "points": len(p["points"])} for p in reference.planes]
    results_path.write_text(json.dumps(results, indent=2))


def depth_maps(args):
    """Score each combination's per-view depth before fusion at its own reference views."""
    tum = Tum(args.dataset)
    summary = json.loads((args.out / "inputs/summary.json").read_text())
    keyframes = {f["frame_index"]: f for f in json.loads(args.keyframes.read_text())["frames"]}
    stride = summary["stride"]
    out = {}
    for combo in summary["combos"]:
        if "measured" in combo:
            continue
        views = np.load(args.out / "inputs" / combo / "views.npz")
        focal = float(views["focal"])
        canonical = combo.startswith("reference")
        counts = dict.fromkeys(metrics.RAY_CLASSES, 0)
        errors = []
        for key in summary["reference_frames"]:
            predicted = views[str(key)]
            rows, cols = predicted.shape
            gy, gx = np.mgrid[0:rows, 0:cols] * stride
            # Truth at the same physical image location as each grid sample.
            truth = tum.kinect(tum.rgb_times[tum.ordinal(keyframes[key])])
            if truth is None:
                continue
            if canonical:
                u, v = sample_tum_pixels(*canonical_rays(gx, gy))
                t = nearest(truth, u, v)
            else:
                t = truth[gy, gx]
            valid = t > 0
            hit = predicted > 0
            error = predicted - t
            counts["missing"] += int((valid & ~hit).sum())
            counts["correct"] += int((valid & hit & (np.abs(error) <= metrics.TOLERANCE_M)).sum())
            counts["false_positive_in_front"] += int((valid & hit & (error < -metrics.TOLERANCE_M)).sum())
            counts["behind_truth"] += int((valid & hit & (error > metrics.TOLERANCE_M)).sum())
            errors.append(np.abs(error[valid & hit]))
        error = np.concatenate(errors)
        out[combo] = {**metrics.ray_summary(counts), "coverage": 1 - metrics.ray_summary(counts)["missing"],
                      "median_abs_depth_error_m": float(np.median(error)), "p90_abs_depth_error_m": float(np.percentile(error, 90))}
        print(combo, json.dumps(out[combo]))
    (args.out / "depth_maps.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------------------------
# Visuals


def heat(error, truth, hit):
    """Signed depth error: blue = surface in front of truth, red = behind; grey = missing."""
    image = np.zeros(error.shape + (3,))
    clipped = np.clip(error / 0.3, -1, 1)
    image[..., 0] = np.where(clipped > 0, 255, 255 * (1 + clipped))
    image[..., 2] = np.where(clipped < 0, 255, 255 * (1 - clipped))
    image[..., 1] = 255 * (1 - np.abs(clipped))
    image[~hit] = (90, 90, 90)
    image[~(truth > 0)] = (0, 0, 0)
    return image.astype(np.uint8)


def visuals(args):
    tum = Tum(args.dataset)
    keyframes = json.loads(args.keyframes.read_text())["frames"]
    evaluation = json.loads(args.evaluation.read_text())["frames"]
    _, held_out = evaluation_ordinals(tum, keyframes, evaluation)
    picks = [int(held_out[round(q * (len(held_out) - 1))]) for q in (0.2, 0.5, 0.8)]
    rays, ys, xs, shape = tum_rays(4, 1)
    rows_spec = [label for label in args.labels]
    panel = (shape[1], shape[0])
    tile_w, tile_h = panel
    columns = 3 * len(picks)
    sheet = Image.new("RGB", (tile_w * columns + 220, tile_h * (len(rows_spec) + 1)), (20, 20, 20))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    truths = []
    for j, ordinal in enumerate(picks):
        time = tum.rgb_times[ordinal]
        pose, depth = tum.pose(time), tum.kinect(time)
        color = tum.color(ordinal)[ys, xs].reshape(shape + (3,))
        truths.append((pose, depth[ys, xs]))
        sheet.paste(Image.fromarray(color), (220 + 3 * j * tile_w, 0))
        d = np.clip(depth[ys, xs] / 4.0, 0, 1).reshape(shape)
        sheet.paste(Image.fromarray((255 * (1 - d) * (d > 0)).astype(np.uint8)).convert("RGB"), (220 + (3 * j + 1) * tile_w, 0))
        draw.text((220 + (3 * j + 2) * tile_w + 4, 4), f"held-out frame {ordinal}\ncols: render |\nshaded | error", fill=(220, 220, 220))
    draw.text((4, 4), "truth: RGB | Kinect", fill=(220, 220, 220))
    for i, label in enumerate(rows_spec):
        path = next(p for p in (args.out / "surfaces").glob(label + ".*") if p.suffix in (".glb", ".ply"))
        surface, _ = load_glb(path, tum, keyframes) if path.suffix == ".glb" else load_ply(path)
        draw.text((4, (i + 1) * tile_h + 4), label.replace("__", "\n"), fill=(220, 220, 220))
        for j, ((r, c), truth) in enumerate(truths):
            depth, _, ray_ids, face_ids, locations = cast(surface, c, r, rays)
            rendered = np.zeros((len(rays), 3))
            rendered[ray_ids] = surface.colorize(face_ids, locations)
            shade = np.zeros(len(rays))
            normals = surface.mesh.face_normals[face_ids]
            shade[ray_ids] = 40 + 200 * np.abs(np.einsum("ij,ij->i", normals, (rays @ r.T)[ray_ids] / np.linalg.norm(rays[ray_ids], axis=1)[:, None]))
            hit = np.isfinite(depth)
            images = [
                np.clip(rendered, 0, 255).reshape(shape + (3,)).astype(np.uint8),
                np.repeat(shade.reshape(shape + (1,)), 3, 2).astype(np.uint8),
                heat(np.nan_to_num(depth - truth).reshape(shape), truth.reshape(shape), hit.reshape(shape)),
            ]
            for k, image in enumerate(images):
                sheet.paste(Image.fromarray(image), (220 + (3 * j + k) * tile_w, (i + 1) * tile_h))
    target = args.out / args.name
    sheet.save(target)
    print(target)


def stage_visuals(args):
    """Estimated depth, validity and its error at three production reference views."""
    tum = Tum(args.dataset)
    summary = json.loads((args.out / "inputs/summary.json").read_text())
    keyframes = {f["frame_index"]: f for f in json.loads(args.keyframes.read_text())["frames"]}
    stride = summary["stride"]
    frames = [summary["reference_frames"][round(q * (len(summary["reference_frames"]) - 1))] for q in (0.2, 0.5, 0.8)]
    views = np.load(args.out / "inputs/estimated_pose__estimated_depth/views.npz")
    from PIL import ImageDraw
    tiles = []
    for key in frames:
        predicted = views[str(key)]
        rows, cols = predicted.shape
        gy, gx = np.mgrid[0:rows, 0:cols] * stride
        truth = tum.kinect(tum.rgb_times[tum.ordinal(keyframes[key])])[gy, gx]
        rgba = np.frombuffer((args.keyframes.parent / keyframes[key]["pixels"]).read_bytes(), np.uint8).reshape(HEIGHT, WIDTH, 4)
        color = rgba[::stride, ::stride, :3]
        d = np.clip(predicted / 4.0, 0, 1)
        estimated = (255 * (1 - d) * (predicted > 0)).astype(np.uint8)
        validity = np.zeros((rows, cols, 3), np.uint8)
        validity[(predicted > 0)] = (60, 200, 90)
        validity[(predicted <= 0) & (truth > 0)] = (220, 60, 60)
        validity[(predicted <= 0) & (truth <= 0)] = (40, 40, 40)
        tiles.append([color, np.repeat(estimated[..., None], 3, 2), validity,
                      heat(predicted - truth, truth, predicted > 0)])
    w, h = cols, rows
    sheet = Image.new("RGB", (w * 4, h * len(tiles) + 16), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 2), "RGB | PatchMatch depth | valid (green) / missing where Kinect valid (red) | error", fill=(230, 230, 230))
    for i, row in enumerate(tiles):
        for k, image in enumerate(row):
            sheet.paste(Image.fromarray(image), (k * w, 16 + i * h))
    sheet.save(args.out / "stage-depth.png")
    print(args.out / "stage-depth.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "learned", "rust", "tsdf", "reference", "colmap", "depth-maps", "evaluate", "visuals", "stage-visuals", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="extracted TUM rgbd_dataset_freiburg3_long_office_household")
    parser.add_argument("--keyframes", type=Path, help="production keyframes.json (108 samples)")
    parser.add_argument("--registered", type=Path, help="accepted COLMAP registered-geometry JSON")
    parser.add_argument("--sparse", type=Path, help="selected COLMAP sparse model")
    parser.add_argument("--stereo", type=Path, help="PatchMatch workspace (reconstruct.py output)")
    parser.add_argument("--grids", type=Path, help="production depth grids (export_grids.py output)")
    parser.add_argument("--evaluation", type=Path, help="evaluation-frame manifest shared across profiles")
    parser.add_argument("--cli", type=Path, help="video-to-3d CLI binary")
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--name", default="comparison.png")
    args = parser.parse_args()
    {"prepare": prepare, "learned": prepare_learned, "rust": run_rust, "tsdf": tsdf, "reference": reference_surface, "colmap": colmap_reference,
     "depth-maps": depth_maps, "evaluate": evaluate, "visuals": visuals, "stage-visuals": stage_visuals,
     "report": report}[args.command](args)


def report(args):
    """Markdown tables and the stage attribution from results.json and depth_maps.json."""
    results = json.loads((args.out / "results.json").read_text())
    depth = json.loads((args.out / "depth_maps.json").read_text())
    rows = []
    for label, entry in results.items():
        if label.startswith("_"):
            continue
        method, pose, kind = label.split("__")
        h, t, p = entry["held_out"], entry["topology"], entry["precision"]
        rows.append({"surface": f"{method} · {pose.replace('_pose', '')} pose · {kind.replace('_depth', '')} depth",
                     **h, **t, **p, "label": label})
    order = ["tsdf", "rust", "prototype-union", "colmap_poisson"]
    rows.sort(key=lambda r: (r["label"].split("__")[2] != "measured_depth", r["label"].split("__")[2],
                             r["label"].split("__")[1], order.index(r["label"].split("__")[0])))
    for row in rows:
        row["area_m2"] = f"{row['area_m2']:.1f}"
        if row.get("held_out_rgb_mae") is not None:
            row["held_out_rgb_mae"] = f"{row['held_out_rgb_mae']:.1f}"
    rays = metrics.markdown_table(rows, [
        ("surface", "Surface", "s"), ("correct", "≤10 cm", "%"), ("correct_within_5cm", "≤5 cm", "%"),
        ("false_positive_in_front", "False positive in front", "%"), ("behind_truth", "Behind truth (hole)", "%"),
        ("missing", "Missing", "%"), ("median_abs_depth_error_m", "Median cm", "cm"), ("p90_abs_depth_error_m", "P90 cm", "cm"),
        ("correct_visible_with_backface_culling", "≤10 cm, culled", "%"), ("held_out_rgb_mae", "RGB MAE", "s")])
    shape = metrics.markdown_table(rows, [
        ("surface", "Surface", "s"), ("triangles", "Triangles", "int"), ("area_m2", "Area m²", "s"),
        ("components", "Components", "int"), ("largest_component_share", "Largest", "%"),
        ("share_in_components_under_32", "In <32-tri pieces", "%"), ("boundary_loops", "Hole/boundary loops", "int"),
        ("precision_within_5cm", "Precision ≤5 cm", "%"), ("free_space_violation", "Free-space violation", "%"),
        ("unverifiable", "Unverifiable", "%")])
    plane_rows = []
    # Planes the 32 reference views never saw (oracle coverage < 5%) cannot discriminate methods.
    oracle = results["tsdf__reference_pose__measured_depth"]["planes"]
    seen = {p["plane"] for p in oracle if p["coverage_within_3cm"] >= 0.05}
    for row in rows:
        for plane in results[row["label"]]["planes"]:
            if plane["plane"] not in seen:
                continue
            plane_rows.append({**plane, "surface": row["surface"], "plane": plane["label"]})
    planes = metrics.markdown_table(plane_rows, [
        ("surface", "Surface", "s"), ("plane", "Reference plane", "s"), ("coverage_within_3cm", "Covered ≤3 cm", "%"),
        ("median_plane_error_m", "Median off-plane cm", "cm"), ("p90_plane_error_m", "P90 off-plane cm", "cm")])
    depth_rows = [{"surface": f"{k.split('__')[1].replace('_depth', '')} depth at {k.split('__')[0].replace('_pose', '')} pose", **v}
                  for k, v in depth.items()]
    maps = metrics.markdown_table(depth_rows, [
        ("surface", "Per-view depth before fusion", "s"), ("correct", "≤10 cm", "%"), ("false_positive_in_front", "In front", "%"),
        ("behind_truth", "Behind", "%"), ("missing", "Missing", "%"),
        ("median_abs_depth_error_m", "Median cm", "cm"), ("p90_abs_depth_error_m", "P90 cm", "cm")])
    correct = {f"{k.split('__')[1]}_{k.split('__')[2]}_{k.split('__')[0]}": results[k]["held_out"]["correct"]
               for k in results if not k.startswith("_")}
    attribution = metrics.attribute_loss(correct)
    text = "\n\n".join(["## Held-out rays", rays, "## Surface shape and precision", shape,
                        "## Per-view depth", maps, "## Reference planes", planes,
                        "## Attribution", "```json\n" + json.dumps(attribution, indent=2) + "\n```"])
    (args.out / "report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
