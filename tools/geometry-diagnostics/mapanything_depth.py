#!/usr/bin/env python3
"""Diagnostic-only MapAnything depth for the TUM ablation's reference views.

Run with the interpreter from scripts/setup_mapanything.sh. Reuses the smoke harness's pinned
inference settings (bf16, memory-efficient inference, minibatch 1) and the estimated COLMAP
calibration as intrinsics. Views are processed in bounded groups and the process is capped below
8 GiB of device memory; peaks are recorded. Output is learned evidence for evaluation, not a
production input: it never crosses the Rust evidence boundary from here.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

MODEL_W, MODEL_H = 504, 378  # Exact 0.7875 scale of 640x480 and a multiple of the 14 px patch.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keyframes", type=Path)
    parser.add_argument("registered", type=Path, help="estimated registration (focal length only)")
    parser.add_argument("summary", type=Path, help="ablation inputs/summary.json (reference frames)")
    parser.add_argument("snapshot", type=Path, help="local MapAnything checkpoint snapshot")
    parser.add_argument("output", type=Path)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--memory-gib", type=float, default=7.5)
    parser.add_argument("--bf16-weights", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    from PIL import Image
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, args.memory_gib * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    manifest = json.loads(args.keyframes.read_text())
    frames = {f["frame_index"]: f for f in manifest["frames"]}
    focal = json.loads(args.registered.read_text())["focal_pixels"] * MODEL_W / 640
    reference = json.loads(args.summary.read_text())["reference_frames"]
    args.output.mkdir(parents=True, exist_ok=True)
    model = MapAnything.from_pretrained(str(args.snapshot)).eval()
    if args.bf16_weights:
        # Halve the resident image encoder and multi-view transformer (the bulk of the weights)
        # when other applications share the 8 GB card. Both already run under bf16 autocast;
        # input encoders and heads that disable autocast stay float32.
        for part in ("encoder", "info_sharing"):
            getattr(model, part).to(dtype=torch.bfloat16)
    model = model.to(device)
    report = {"device": torch.cuda.get_device_name(device), "memory_cap_bytes": int(fraction * total),
              "weights_dtype": "bfloat16 encoder+transformer" if args.bf16_weights else "float32",
              "calibration_input": not args.bf16_weights,
              "device_free_after_load_bytes": torch.cuda.mem_get_info(device)[0],
              "model_size": [MODEL_W, MODEL_H], "group": args.group, "groups": []}
    for start in range(0, len(reference), args.group):
        group = reference[start:start + args.group]
        torch.cuda.reset_peak_memory_stats(device)
        began = time.monotonic()
        with tempfile.TemporaryDirectory() as scratch:
            paths = []
            for frame in group:
                record = frames[frame]
                rgba = np.frombuffer((args.keyframes.parent / record["pixels"]).read_bytes(), np.uint8)
                image = Image.fromarray(rgba.reshape(record["height"], record["width"], 4)[..., :3])
                path = Path(scratch) / f"{frame:08d}.png"
                image.resize((MODEL_W, MODEL_H), Image.Resampling.BICUBIC).save(path)
                paths.append(str(path))
            views = load_images(paths, resize_mode="fixed_size", size=(MODEL_W, MODEL_H))
            k = torch.tensor([[focal, 0, MODEL_W / 2], [0, focal, MODEL_H / 2], [0, 0, 1]], dtype=torch.float32)
            for view in views:
                if tuple(view["img"].shape[-2:]) != (MODEL_H, MODEL_W):
                    raise ValueError(f"unexpected model input shape {tuple(view['img'].shape)}")
                # Upstream fuses calibration rays into bf16 encoder features without a cast, so
                # half-precision weights run image-only; the predicted z-depth is used per pixel.
                if not args.bf16_weights:
                    view["intrinsics"] = k[None]
            with torch.inference_mode():
                predictions = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                          use_amp=True, amp_dtype="bf16", apply_mask=True, mask_edges=True)
        for frame, prediction in zip(group, predictions):
            depth = prediction["depth_z"][0].squeeze(-1).float().cpu().numpy()
            mask = prediction["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & np.isfinite(depth) & (depth > 0)
            confidence = prediction["conf"][0].float().cpu().numpy() if "conf" in prediction else np.ones_like(depth)
            np.savez_compressed(args.output / f"{frame}.npz", depth=np.where(mask, depth, 0).astype(np.float32),
                                confidence=confidence.astype(np.float32).squeeze())
        report["groups"].append({
            "frames": group, "elapsed_seconds": round(time.monotonic() - began, 2),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        })
        print(json.dumps(report["groups"][-1]))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
