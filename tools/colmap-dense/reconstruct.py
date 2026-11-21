# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["pycolmap-cuda12==4.2.1"]
# ///
"""Run geometric PatchMatch and multiview fusion on a local COLMAP sparse model."""
import argparse
import json
from pathlib import Path
from time import monotonic

import pycolmap


def select_sources(model, limit):
    images = sorted(model.images.values(), key=lambda im: im.name)
    tracks = {im.name: {p.point3D_id for p in im.points2D if p.has_point3D()} for im in images}
    neighbors = {
        name: sorted(((len(points & other), key) for key, other in tracks.items() if key != name),
                     key=lambda pair: (-pair[0], pair[1]))
        for name, points in tracks.items()
    }
    if limit >= len(images):
        selected = set(tracks)
    else:
        # Spread anchors along the video. For each anchor retain a small covisible cluster,
        # so PatchMatch has actual neighboring depth maps for its geometric consistency pass.
        selected = set()
        anchors = max(1, limit // 4)
        for i in range(anchors):
            index = round((len(images) - 1) * (i + 0.5) / anchors)
            anchor = images[index].name
            selected.add(anchor)
            selected.update(key for count, key in neighbors[anchor][:3] if count >= 8)
    config = {}
    for name in sorted(selected):
        sources = [key for count, key in neighbors[name] if key in selected and count >= 8][:4]
        if len(sources) >= 2:
            config[name] = sources
    while any(any(source not in config for source in sources) for sources in config.values()):
        config = {name: [source for source in sources if source in config] for name, sources in config.items()}
        config = {name: sources for name, sources in config.items() if len(sources) >= 2}
    return config


def reconstruct(images, sparse, output, max_image_size, max_views):
    if not pycolmap.has_cuda or pycolmap.get_num_cuda_devices() < 1:
        raise RuntimeError("CUDA support is required for COLMAP PatchMatch")
    model = pycolmap.Reconstruction(sparse)
    if model.num_reg_images() < 3:
        raise ValueError("geometric stereo needs at least three registered views")
    if max_views < 4:
        raise ValueError("need at least four stereo views")
    if max_image_size < 1:
        raise ValueError("max image size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    start = monotonic()
    config = select_sources(model, max_views)
    if len(config) < 3:
        raise ValueError("insufficient covisible camera groups for geometric stereo")
    pycolmap.undistort_images(
        output, sparse, images, image_names=list(config), num_threads=2, num_patch_match_src_images=4,
        undistort_options=pycolmap.UndistortCameraOptions(max_image_size=max_image_size),
    )
    stereo = output / "stereo"
    (stereo / "patch-match.cfg").write_text("".join(name + "\n" + ", ".join(sources) + "\n" for name, sources in config.items()))
    (stereo / "fusion.cfg").write_text("".join(name + "\n" for name in config))
    pycolmap.patch_match_stereo(
        output, options=pycolmap.PatchMatchOptions(
            gpu_index="0", geom_consistency=True, num_threads=2, cache_size=1.0,
        ),
    )
    fused = pycolmap.stereo_fusion(
        output / "fused.ply", output, input_type="geometric", output_type="ply",
        options=pycolmap.StereoFusionOptions(num_threads=1, cache_size=1.0),
    )
    summary = {
        "provider": "pycolmap-cuda12", "version": pycolmap.__version__,
        "registered_images": model.num_reg_images(), "sparse_points": model.num_points3D(),
        "fused_points": fused.num_points3D(), "max_image_size": max_image_size,
        "stereo_views": len(config), "max_source_views": 4, "elapsed_seconds": round(monotonic() - start, 3),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not fused.num_points3D():
        raise ValueError("no observations survived geometric stereo fusion")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("sparse", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-image-size", type=int, default=640)
    parser.add_argument("--max-views", type=int, default=32)
    print(json.dumps(reconstruct(**vars(parser.parse_args()))))
