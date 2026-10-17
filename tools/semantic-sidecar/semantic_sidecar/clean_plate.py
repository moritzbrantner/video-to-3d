"""Local LaMa (big-lama) clean-plate sidecar (protocol video-to-3d-clean-plate-sidecar/1).

Invoked by the Rust `clean_plate` operation as

    video-to-3d-clean-plate-sidecar <request.json> <response.json>

Each edit names a local keyframe (raw RGBA8), a row-major run-length removal
mask and an output path. The sidecar inpaints the masked pixels with LaMa
(Atlas 8.9) and writes the edited RGBA8 frame; pixels outside the mask are
kept exactly. Weights (the pinned big-lama TorchScript release) are fetched
once into the local cache on first need. Runs on CUDA when available, else
on the CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROTOCOL = "video-to-3d-clean-plate-sidecar/1"
MODEL_ID = "big-lama (TorchScript, simple-lama-inpainting v0.1.0 release)"
MODEL_VERSION = "v0.1.0"
MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


def cached_model() -> Path:
    import torch

    cache = Path(torch.hub.get_dir()) / "video-to-3d"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "big-lama-v0.1.0.pt"
    if not path.exists():
        partial = path.with_suffix(".part")
        torch.hub.download_url_to_file(MODEL_URL, str(partial), progress=False)
        partial.rename(path)
    return path


def decode(counts: list[int], width: int, height: int):
    import numpy as np

    values = np.zeros(width * height, dtype=bool)
    position = 0
    for index, run in enumerate(counts):
        if index % 2 == 1:
            values[position : position + run] = True
        position += run
    return values.reshape(height, width)


def main(request_path: str, response_path: str) -> None:
    request = json.loads(Path(request_path).read_text())
    if request.get("protocol") != PROTOCOL:
        raise SystemExit(f"unsupported protocol {request.get('protocol')!r}")
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(str(cached_model()), map_location=device).eval()
    results = []
    with torch.inference_mode():
        for edit in request["edits"]:
            try:
                width, height = edit["width"], edit["height"]
                rgba = np.frombuffer(Path(edit["rgba_path"]).read_bytes(), dtype=np.uint8).reshape(height, width, 4)
                mask = decode(edit["mask_counts"], width, height)
                # LaMa wants sizes divisible by 8: pad by reflection.
                pad_h, pad_w = (-height) % 8, (-width) % 8
                image = np.pad(rgba[:, :, :3], ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                hole = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
                image_t = torch.from_numpy(image).permute(2, 0, 1)[None].float().div(255).to(device)
                mask_t = torch.from_numpy(hole.astype(np.float32))[None, None].to(device)
                result = model(image_t, mask_t)[0].permute(1, 2, 0).clamp(0, 1).mul(255).round()
                result = result.byte().cpu().numpy()[:height, :width]
                output = rgba.copy()
                output[mask, :3] = result[mask]
                Path(edit["output_path"]).write_bytes(output.tobytes())
                results.append({"edit_id": edit["edit_id"], "status": "ok"})
            except Exception as error:  # reported per edit; others continue
                results.append({"edit_id": edit["edit_id"], "status": "failed", "reason": f"{type(error).__name__}: {error}"})
    Path(response_path).write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": "available",
                "models": [{"role": "inpainter", "id": MODEL_ID, "version": MODEL_VERSION, "url": MODEL_URL}],
                "runtime": {
                    "name": "pytorch",
                    "version": torch.__version__,
                    "device": device,
                    "device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
                },
                "edits": results,
            }
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: video-to-3d-clean-plate-sidecar <request.json> <response.json>")
    main(sys.argv[1], sys.argv[2])
