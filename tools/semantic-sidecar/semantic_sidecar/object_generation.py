"""Local TripoSR object-generation sidecar (protocol video-to-3d-object-generation-sidecar/1).

Invoked by the Rust `object_generation` operation as

    video-to-3d-object-generation-sidecar <request.json> <response.json>

Each request names a local keyframe (raw RGBA8), a crop box and the object's
exact row-major run-length mask. The sidecar cuts the object out, composites
it on TripoSR's neutral grey background, reconstructs a mesh with TripoSR
(Atlas 8.10) and writes it as binary glTF. The TripoSR code (MIT) is fetched
at a pinned commit and its weights from the Hugging Face cache on first need;
marching cubes uses scikit-image instead of the compiled `torchmcubes`.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import types
import urllib.request
from pathlib import Path

PROTOCOL = "video-to-3d-object-generation-sidecar/1"
TRIPOSR_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"
TRIPOSR_SOURCE = f"https://github.com/VAST-AI-Research/TripoSR/archive/{TRIPOSR_COMMIT}.tar.gz"
MODEL_ID = "stabilityai/TripoSR"
RESOLUTION = 256
FOREGROUND_RATIO = 0.85


def triposr_source() -> Path:
    """The pinned TripoSR checkout in the local cache (fetched once)."""
    import torch

    root = Path(torch.hub.get_dir()) / "video-to-3d" / f"TripoSR-{TRIPOSR_COMMIT}"
    if not (root / "tsr" / "system.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TRIPOSR_SOURCE, timeout=120) as response:
            data = response.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            archive.extractall(root.parent, filter="data")
    return root


def install_shims() -> None:
    """Replace TripoSR's compiled/unneeded imports with local equivalents."""
    import numpy as np
    import torch
    from skimage import measure

    def marching_cubes(volume, threshold):
        values = volume.detach().float().cpu().numpy()
        vertices, faces, _, _ = measure.marching_cubes(values, level=threshold)
        # torchmcubes returns (x, y, z) for a volume indexed [z][y][x].
        vertices = np.ascontiguousarray(vertices[:, ::-1])
        faces = np.ascontiguousarray(faces[:, ::-1])
        return torch.from_numpy(vertices.copy()).float(), torch.from_numpy(faces.astype(np.int64))

    sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=marching_cubes)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("background removal is not used: masks come from the semantic inventory")

    sys.modules["rembg"] = types.SimpleNamespace(remove=unavailable, new_session=unavailable)


def decode(counts, width, height):
    import numpy as np

    values = np.zeros(width * height, dtype=bool)
    position = 0
    for index, run in enumerate(counts):
        if index % 2 == 1:
            values[position : position + run] = True
        position += run
    return values.reshape(height, width)


def object_image(entry):
    """Masked object crop composited on grey, square and centred (TripoSR input)."""
    import numpy as np
    from PIL import Image

    width, height = entry["width"], entry["height"]
    rgba = np.frombuffer(Path(entry["rgba_path"]).read_bytes(), dtype=np.uint8).reshape(height, width, 4).copy()
    mask = decode(entry["mask_counts"], width, height)
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    x0, y0, x1, y1 = entry["crop"]
    crop = rgba[y0:y1, x0:x1]
    alpha = np.where(crop[..., 3] > 0)
    if alpha[0].size == 0:
        raise ValueError("the mask is empty inside the crop")
    crop = crop[alpha[0].min() : alpha[0].max() + 1, alpha[1].min() : alpha[1].max() + 1]
    side = int(max(crop.shape[:2]) / FOREGROUND_RATIO)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    top, left = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
    image = canvas.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4]) * 0.5
    return Image.fromarray((image * 255.0).astype(np.uint8)).resize((512, 512))


def main(request_path: str, response_path: str) -> None:
    request = json.loads(Path(request_path).read_text())
    if request.get("protocol") != PROTOCOL:
        raise SystemExit(f"unsupported protocol {request.get('protocol')!r}")
    import torch

    if not torch.cuda.is_available():
        Path(response_path).write_text(json.dumps({
            "protocol": PROTOCOL, "status": "unavailable",
            "reason": "no CUDA device is available; local object generation needs a GPU, so reconstructed geometry stays the only object evidence",
        }))
        return
    device = "cuda"
    install_shims()
    sys.path.insert(0, str(triposr_source()))
    from tsr.system import TSR

    model = TSR.from_pretrained(MODEL_ID, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)
    version = getattr(getattr(model, "cfg", None), "_commit_hash", None) or "model.ckpt"
    results = []
    for entry in request["requests"]:
        try:
            image = object_image(entry)
            with torch.no_grad():
                scene_codes = model([image], device=device)
                mesh = model.extract_mesh(scene_codes, True, resolution=RESOLUTION)[0]
            mesh.export(entry["output_path"], file_type="glb")
            results.append({
                "request_id": entry["request_id"], "status": "ok",
                "vertices": int(len(mesh.vertices)), "triangles": int(len(mesh.faces)),
            })
        except Exception as error:  # reported per object; others continue
            results.append({"request_id": entry["request_id"], "status": "failed", "reason": f"{type(error).__name__}: {error}"})
        torch.cuda.empty_cache()
    Path(response_path).write_text(json.dumps({
        "protocol": PROTOCOL,
        "status": "available",
        "models": [{"role": "generator", "id": MODEL_ID, "version": str(version),
                    "url": f"https://huggingface.co/{MODEL_ID}"},
                   {"role": "generator", "id": "VAST-AI-Research/TripoSR (code)", "version": TRIPOSR_COMMIT,
                    "url": TRIPOSR_SOURCE}],
        "runtime": {"name": "pytorch", "version": torch.__version__, "device": device,
                    "device_name": torch.cuda.get_device_name(0)},
        "results": results,
    }))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: video-to-3d-object-generation-sidecar <request.json> <response.json>")
    main(sys.argv[1], sys.argv[2])
