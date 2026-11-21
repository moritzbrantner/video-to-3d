"""Register local RGBA keyframes. Output feature tracks for independent Rust validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colmap_golden import solve
from tracks import choose_exportable_seed
from revisits import revisit_pairs
from global_recovery import recover_globally


def register(manifest, work):
    frames = json.loads(manifest.read_text())["frames"]
    if len(frames) < 3:
        raise ValueError("COLMAP registration requires at least three sampled frames")
    images = work / "images"
    images.mkdir()
    frame_hashes = []
    width, height = frames[0]["width"], frames[0]["height"]
    for index, frame in enumerate(frames):
        if frame["frame_index"] != index or (frame["width"], frame["height"]) != (width, height):
            raise ValueError("inconsistent keyframe dimensions or numbering")
        path = (manifest.parent / frame["pixels"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError("keyframe path outside manifest directory")
        data = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if digest != frame["content_hash"] or len(data) != width * height * 4:
            raise ValueError("keyframe data differs from its receipt")
        frame_hashes.append(dict(frame_index=index, content_hash=digest))
        Image.frombytes("RGBA", (width, height), data).convert("RGB").save(images / f"{index:08}.png")

    initial_focal = max(width, height) * 0.86
    cpu_count = os.cpu_count() or 1
    diagnostics = {}
    maps = solve(
        images, work, f"{initial_focal},{width / 2},{height / 2}", refine_focal=True,
        feature_threads=min(cpu_count, 4), match_threads=min(cpu_count, 6),
        # Bound the first pass on long, detailed recordings. The mapper still
        # tries all pairs when temporal overlaps cannot connect the whole clip.
        exhaustive_comparison_budget=4_000_000_000, propose_revisits=revisit_pairs,
        recover_partial=recover_globally, diagnostics=diagnostics,
    )
    if not maps:
        raise ValueError("COLMAP did not register three overlapping views")
    model_id, model = max(maps.items(), key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()))
    if model.num_reg_images() < 3:
        raise ValueError("COLMAP did not register three overlapping views")
    registered = sorted((model.images[key] for key in model.reg_image_ids()), key=lambda i: i.name)
    camera = model.cameras[registered[0].camera_id]
    first, second, unambiguous_ids = choose_exportable_seed(
        model, registered, check_global_parallax=diagnostics.get("mapper") == "global"
    )
    ordered = [first, second] + [i for i in registered if i.image_id not in (first.image_id, second.image_id)]
    origin = first.projection_center()
    basis = first.cam_from_world().rotation.matrix()
    scale = np.linalg.norm(second.projection_center() - origin)
    if scale <= 1e-9:
        raise ValueError("stationary camera path")
    cameras = []
    frames_by_image = {i.image_id:int(Path(i.name).stem) for i in registered}
    for image in ordered:
        r = image.cam_from_world().rotation.matrix() @ basis.T
        t = r @ (basis @ (origin - image.projection_center())) / scale
        cameras.append(dict(frame_index=frames_by_image[image.image_id],rotation=r.ravel().tolist(),translation=t.tolist()))
    landmarks = []
    for point_id in unambiguous_ids:
        point = model.points3D[point_id]
        observations = [dict(frame_index=frames_by_image[e.image_id],xy=model.images[e.image_id].points2D[e.point2D_idx].xy.tolist()) for e in point.track.elements]
        landmarks.append(dict(xyz=(basis @ (point.xyz-origin)/scale).tolist(), rgb=point.color.tolist(), observations=observations))
    (work / "selected-model.txt").write_text(f"sparse/{model_id}\n", encoding="utf-8")
    print(
        f"COLMAP registered {len(cameras)}/{len(frames)} frames in the largest of {len(maps)} "
        f"model(s), with {len(landmarks)} unambiguous tracks and "
        f"{model.compute_mean_reprojection_error():.2f}px mean model reprojection error.",
        file=sys.stderr,
    )
    diagnostics.update(registered_frames=len(cameras), input_frames=len(frames))
    return dict(schema_version=1,provider="colmap",version=pycolmap.__version__,frames=frame_hashes,
                registration_diagnostics=diagnostics,
                width=width,height=height,focal_pixels=float(camera.params[0]),cameras=cameras,landmarks=landmarks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("workdir", type=Path, nargs="?", help="keep the input images and selected sparse model for dense reconstruction")
    args = parser.parse_args()
    if args.workdir is None:
        with TemporaryDirectory(prefix="video-to-3d-colmap-") as scratch:
            result = register(args.manifest, Path(scratch))
    else:
        args.workdir.mkdir(parents=True, exist_ok=True)
        if any(args.workdir.iterdir()):
            parser.error("COLMAP workdir must be empty for a fresh solve")
        result = register(args.manifest, args.workdir)
    args.output.write_text(json.dumps(result, separators=(",", ":")))
