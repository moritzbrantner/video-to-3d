# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["pycolmap==4.2.1"]
# ///
"""Sample geometric depth into the verified sparse provider's image grid and unit-baseline gauge."""
import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap


def export_grids(registered, sparse, workspace, output):
    registration = json.loads(registered.read_text())
    model = pycolmap.Reconstruction(sparse)
    dense = pycolmap.Reconstruction(workspace / "sparse")
    images = {int(Path(i.name).stem): i for i in model.images.values()}
    cameras = registration["cameras"]
    first, second = [images[c["frame_index"]] for c in cameras[:2]]
    origin = first.projection_center()
    basis = first.cam_from_world().rotation.matrix()
    scale = np.linalg.norm(second.projection_center() - origin)
    if not scale > 1e-9 or len(cameras) != len(images):
        raise ValueError("dense workspace and sparse camera registration differ")
    width, height = registration["width"], registration["height"]
    # Verify a projected depth on a finer source grid than the rendered mesh.
    # This keeps the reference topology bounded while resolving more source-side edges.
    reference_stride = max(4, (min(width, height) // 56) // 2 * 2)
    stride = reference_stride // 2
    y, x = np.mgrid[0:height:stride, 0:width:stride]
    grid = np.column_stack((x.ravel() + 0.5, y.ravel() + 0.5))
    tracks = {n: {p.point3D_id for p in im.points2D if p.has_point3D()} for n, im in images.items()}
    measured = {path.name.removesuffix(".geometric.bin") for path in (workspace / "stereo/depth_maps").glob("*.geometric.bin")}
    maps = []
    for c in sorted(cameras, key=lambda c: c["frame_index"]):
        n = c["frame_index"]
        im = images[n]
        if im.name not in measured:
            continue
        source = model.cameras[im.camera_id]
        rot = im.cam_from_world().rotation.matrix() @ basis.T
        trans = rot @ basis @ (origin - im.projection_center()) / scale
        if not np.allclose(rot.ravel(), c["rotation"], atol=1e-6) or not np.allclose(trans, c["translation"], atol=1e-6):
            raise ValueError("dense map registration gauge differs from verified sparse poses")
        target = dense.cameras[dense.images[im.image_id].camera_id]
        plane = source.cam_from_img(grid)
        xy = np.floor(target.img_from_cam(np.column_stack((plane, np.ones(len(grid)))))).astype(int)
        u, v = xy.T
        depth = pycolmap.DepthMap()
        depth.read(workspace / "stereo/depth_maps" / (im.name + ".geometric.bin"))
        raw = depth.to_array()
        ok = (u >= 0) & (v >= 0) & (u < raw.shape[1]) & (v < raw.shape[0])
        z = np.zeros(len(grid), dtype='f4')
        z[ok] = raw[v[ok], u[ok]] / scale
        z[~np.isfinite(z) | (z <= 0)] = 0
        others = sorted((k for k in tracks if k != n and images[k].name in measured), key=lambda k: (-len(tracks[n] & tracks[k]), k))
        maps.append(dict(frame=n, sources=[k for k in others[:4] if len(tracks[n] & tracks[k]) >= 8], depth=z.astype(float).round(6).tolist()))
    with output.open("w") as f:
        json.dump(dict(version=1, stride=stride, reference_stride=reference_stride, maps=maps), f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registered", "sparse", "workspace", "output"):
        parser.add_argument(name, type=Path)
    export_grids(**vars(parser.parse_args()))
