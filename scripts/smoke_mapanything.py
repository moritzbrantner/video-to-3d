#!/usr/bin/env python3
"""Reproducible, sequential MapAnything GPU smoke runs on browser-prepared frames.

The parent uses only the Python standard library. Model dependencies are needed
only in the selected Python environment when the child process starts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib


MODEL_ID = "facebook/map-anything-apache"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
REPORT_SCHEMA = "video-to-3d/mapanything-gpu-smoke/v1"


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def rgba_to_png(rgba: bytes, width: int, height: int) -> bytes:
    """Losslessly convert opaque browser RGBA to RGB PNG without another decoder."""
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise ValueError("invalid browser frame dimensions")
    if len(rgba) != width * height * 4:
        raise ValueError("browser frame byte length does not match dimensions")
    if any(alpha != 255 for alpha in rgba[3::4]):
        raise ValueError("browser frame has transparency; refusing ambiguous RGB conversion")
    stride = width * 4
    scanlines = bytearray()
    for offset in range(0, len(rgba), stride):
        scanlines.append(0)  # PNG no-filter scanline
        row = rgba[offset : offset + stride]
        for pixel in range(0, stride, 4):
            scanlines.extend(row[pixel : pixel + 3])
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6))
        + png_chunk(b"IEND", b"")
    )


def read_capture(root: Path, frame_dir: Path) -> tuple[list[Path], list[dict]]:
    root = root.resolve(strict=True)
    document = json.loads((root / "browser-frames.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("frames"), list):
        raise ValueError("unsupported browser-frames.json schema")
    if len(document["frames"]) < 2:
        raise ValueError("at least two browser frames are required")
    images, sources = [], []
    frame_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, record in enumerate(document["frames"]):
        if record.get("frame_index") != ordinal:
            raise ValueError("browser frame order/index mismatch")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ValueError("missing browser frame path")
        source = (root / relative).resolve(strict=True)
        if not source.is_relative_to(root) or source.suffix != ".rgba":
            raise ValueError("browser frame path escapes the capture root")
        raw = source.read_bytes()
        if len(raw) != record.get("byte_length") or digest(raw) != record.get("content_hash"):
            raise ValueError(f"browser frame {ordinal} failed its byte/hash receipt")
        target = frame_dir / f"{ordinal:04d}.png"
        target.write_bytes(rgba_to_png(raw, record["width"], record["height"]))
        images.append(target)
        sources.append(
            {
                "index": ordinal,
                "source_sha256": digest(raw),
                "derived_png_sha256": digest(target.read_bytes()),
                "width": record["width"],
                "height": record["height"],
                "presented_time": record.get("presented_time"),
            }
        )
    return images, sources


def read_images(root: Path) -> tuple[list[Path], list[dict]]:
    images = sorted(
        path
        for path in root.resolve(strict=True).iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    )
    if len(images) < 2:
        raise ValueError("at least two pre-decoded RGB images are required")
    return images, [{"index": i, "name": p.name, "source_sha256": digest(p.read_bytes())} for i, p in enumerate(images)]


def select_indices(total: int, count: int, start: int = 0, policy: str = "contiguous") -> list[int]:
    if start < 0 or count < 2 or count > total - start:
        raise ValueError("view count must fit the available frames after the start index")
    if policy == "contiguous":
        return list(range(start, start + count))
    if policy == "uniform":
        return [start + i * (total - start - 1) // (count - 1) for i in range(count)]
    raise ValueError("unknown view selection policy")


def git_head(directory: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def hardware() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"nvidia_smi": "unavailable"}
    return {"nvidia_smi": result.stdout.strip() if result.returncode == 0 else "unavailable"}


def device_used_mib() -> int | None:
    """Sample total used memory on physical GPU 0 (includes other processes)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return int(result.stdout.strip().splitlines()[0].strip()) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
        return None


def terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def child(command: list[str], output: Path, timeout: int, env: dict, measure_gpu: bool = False) -> tuple[int, float, dict]:
    start = time.monotonic()
    baseline_mib = device_used_mib() if measure_gpu else None
    peak_mib = baseline_mib
    with (
        (output / "stdout.log").open("w", encoding="utf-8") as stdout,
        (output / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            while True:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    terminate_group(process)
                    code = 124
                    break
                try:
                    code = process.wait(timeout=min(0.5, timeout - elapsed))
                except subprocess.TimeoutExpired:
                    code = None
                if measure_gpu:
                    sample = device_used_mib()
                    if sample is not None:
                        peak_mib = max(peak_mib or 0, sample)
                if code is not None:
                    break
        except KeyboardInterrupt:
            terminate_group(process)
            raise
    measurements = {
        "nvidia_smi_gpu0_baseline_used_mib": baseline_mib,
        "nvidia_smi_gpu0_peak_used_sampled_mib": peak_mib,
        "nvidia_smi_gpu0_peak_increase_sampled_mib": (
            peak_mib - baseline_mib if baseline_mib is not None and peak_mib is not None else None
        ),
        "nvidia_smi_sample_interval_seconds": 0.5 if measure_gpu else None,
    } if measure_gpu else {}
    return code, round(time.monotonic() - start, 3), measurements


def worker_resolve(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=args.revision, local_files_only=args.offline)).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
        raise RuntimeError("Hugging Face did not return a commit-addressed snapshot")
    write_json(
        Path(args.result),
        {
            "model_id": MODEL_ID,
            "revision": snapshot.name,
            "snapshot_path": str(snapshot),
            "requested_revision": args.revision,
        },
    )
    return 0


def worker_infer(args: argparse.Namespace) -> int:
    # Import model dependencies only inside the disposable GPU process.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.image import load_images
    from mapanything.utils.viz import predictions_to_glb
    import mapanything

    output = Path(args.result)
    if not torch.cuda.is_available():
        write_json(output, {"status": "unavailable", "reason": "CUDA PyTorch is unavailable"})
        return 3
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    total = torch.cuda.get_device_properties(device).total_memory
    free_before, _ = torch.cuda.mem_get_info(device)
    started = time.monotonic()
    from importlib.metadata import version

    report = {
        "device": torch.cuda.get_device_name(device),
        "device_total_bytes": total,
        "device_free_before_bytes": free_before,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mapanything_version": version("mapanything"),
        "mapanything_git_commit": git_head(Path(mapanything.__file__).resolve().parents[1]),
        "checkpoint_commit": Path(args.snapshot).name,
    }
    try:
        model = MapAnything.from_pretrained(args.snapshot).to(device).eval()
        views = load_images(json.loads(Path(args.images_json).read_text(encoding="utf-8")))
        report["model_input_tensor_shapes"] = [list(view["img"].shape) for view in views]
        with torch.inference_mode():
            predictions = model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
            )
        torch.cuda.synchronize(device)
        cameras, world_points, colors, masks = [], [], [], []
        for prediction in predictions:
            depth = prediction["depth_z"][0].squeeze(-1)
            intrinsics = prediction["intrinsics"][0]
            pose = prediction["camera_poses"][0]
            points, valid_depth = depthmap_to_world_frame(depth, intrinsics, pose)
            mask = prediction["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & valid_depth.cpu().numpy()
            mask &= torch.isfinite(points).all(dim=-1).cpu().numpy()
            if not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(intrinsics).all()) or not mask.any():
                raise ValueError("model emitted nonfinite camera parameters or no valid geometry")
            cameras.append(
                {
                    "intrinsics": intrinsics.cpu().tolist(),
                    "camera_to_world": pose.cpu().tolist(),
                    "valid_points": int(mask.sum()),
                    "total_points": int(mask.size),
                }
            )
            if args.export_glb:
                world_points.append(points.cpu().numpy())
                colors.append(prediction["img_no_norm"][0].cpu().numpy())
                masks.append(mask)
        report.update(
            {"status": "model_completed", "cameras": cameras, "view_count": len(predictions), "preview_exported": False}
        )
        if args.export_glb:
            try:
                scene = predictions_to_glb(
                    {"world_points": np.stack(world_points), "images": np.stack(colors), "final_masks": np.stack(masks)},
                    as_mesh=True,
                )
                scene.export(str(output.parent / "preview.glb"))
                report["preview_exported"] = True
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as error:
                import traceback

                traceback.print_exc()
                report["status"] = "preview_failed"
                report["preview_error"] = f"{type(error).__name__}: {error}"[:500]
        returncode = 0 if report["status"] == "model_completed" else 5
    except torch.cuda.OutOfMemoryError as error:
        report.update({"status": "oom", "reason": str(error)[:500]})
        returncode = 42
    except Exception as error:
        import traceback

        traceback.print_exc()
        report.update({"status": "failed", "reason": f"{type(error).__name__}: {error}"[:500]})
        returncode = 1
    report.update(
        {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_free_after_bytes": torch.cuda.mem_get_info(device)[0],
        }
    )
    write_json(output, report)
    return returncode


def zip_logs(root: Path) -> None:
    with zipfile.ZipFile(root / "share-logs.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".log"}:
                if path.name == "checkpoint.json":
                    # The shareable archive does not need the machine-local HF cache path.
                    continue
                archive.write(path, path.relative_to(root))


def can_reuse_report(root: Path, plan: dict, revision: str) -> bool:
    """Mutable model branches cannot be reused as an immutable GPU measurement."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        return False
    try:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        summary.get("plan") == plan
        and summary.get("status") == "completed"
        and summary.get("checkpoint_commit") == revision
        and len(summary.get("cases", [])) == len(plan["cases"])
        and all(case.get("status") == "model_completed" for case in summary["cases"])
        and (root / "share-logs.zip").is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--capture-root", type=Path, help="browser-frames.json and .rgba frames produced by the existing browser capture"
    )
    sources.add_argument("--images-dir", type=Path, help="already-decoded local images (never decodes the original video)")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh or previously completed result directory")
    parser.add_argument(
        "--view-counts", default="8",
        help="view counts, comma-separated and increasing (default: 8; use 8,12 to compare)",
    )
    parser.add_argument("--start-index", type=int, default=0, help="first usable frame index, default 0")
    parser.add_argument(
        "--sampling-policy",
        choices=["contiguous", "uniform"],
        default="contiguous",
        help="contiguous frames by default; uniform spans the remainder of the video",
    )
    parser.add_argument(
        "--revision", default="main", help="Hugging Face model commit or branch; resolved commit is recorded"
    )
    parser.add_argument("--offline", action="store_true", help="use only already-cached checkpoint files")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="per-model-inference timeout")
    parser.add_argument("--no-glb", action="store_true", help="skip rendering the model-only GLB preview")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate inputs and print the planned samples without CUDA/model imports"
    )
    parser.add_argument("--_worker", choices=["resolve", "infer"], help=argparse.SUPPRESS)
    parser.add_argument("--_result", help=argparse.SUPPRESS)
    parser.add_argument("--_snapshot", help=argparse.SUPPRESS)
    parser.add_argument("--_images_json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        wargs = argparse.Namespace(
            revision=args.revision,
            offline=args.offline,
            result=args._result,
            snapshot=args._snapshot,
            images_json=args._images_json,
            export_glb=not args.no_glb,
        )
        return worker_resolve(wargs) if args._worker == "resolve" else worker_infer(wargs)
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    try:
        counts = [int(part) for part in args.view_counts.split(",")]
        if not counts or counts != sorted(set(counts)) or any(count < 2 for count in counts):
            raise ValueError("view counts must be unique, increasing, and >= 2")
        with tempfile.TemporaryDirectory(prefix="video-to-3d-model-smoke-") as temp:
            images, sources = (
                read_capture(args.capture_root, Path(temp) / "images")
                if args.capture_root
                else read_images(args.images_dir)
            )
            plans = [
                {
                    "views": count,
                    "source_indices": select_indices(len(images), count, args.start_index, args.sampling_policy),
                }
                for count in counts
            ]
            plan = {
                "schema": REPORT_SCHEMA,
                "model_id": MODEL_ID,
                "revision": args.revision,
                "input_frames": sources,
                "cases": plans,
                "sampling_policy": args.sampling_policy,
                "start_index": args.start_index,
                "input_type": "browser_capture" if args.capture_root else "decoded_images",
                "inference": {
                    "memory_efficient_inference": True,
                    "minibatch_size": 1,
                    "use_amp": True,
                    "amp_dtype": "bf16",
                    "apply_mask": True,
                    "mask_edges": True,
                },
                "export_preview_glb": not args.no_glb,
                "upstream_geometry_accepted_by_rust": False,
            }
            if args.dry_run:
                print(json.dumps(plan, indent=2))
                return 0
            root = args.output_dir.resolve()
            if root.exists():
                prior = root / "summary.json"
                if can_reuse_report(root, plan, args.revision):
                    print(f"Existing report reused: {prior}")
                    return 0
                raise ValueError(
                    "output directory is incomplete, differs, or uses a mutable model revision; use a fresh directory"
                )
            root.mkdir(parents=True)
            write_json(root / "plan.json", plan)
            gpu = hardware()
            write_json(
                root / "hardware.json",
                {
                    "gpu": gpu,
                    "python": sys.version.split()[0],
                    "video_to_3d_commit": git_head(Path(__file__).resolve().parents[1]),
                },
            )
            if gpu["nvidia_smi"] == "unavailable":
                write_json(
                    root / "summary.json",
                    {
                        "plan": plan,
                        "status": "gpu_unavailable",
                        "reason": "nvidia-smi failed; refusing to download the checkpoint without an NVIDIA GPU",
                    },
                )
                zip_logs(root)
                print(f"NVIDIA GPU unavailable; diagnostics: {root / 'share-logs.zip'}")
                return 1
            env = dict(os.environ)
            lock_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "video-to-3d"
            lock_root.mkdir(parents=True, exist_ok=True)
            with (lock_root / "local-gpu-model-smoke.lock").open("a+") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError("another video-to-3d GPU smoke run owns the GPU lock") from error
                resolve_dir = root / "resolve"
                resolve_dir.mkdir()
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--images-dir",
                    str(args.images_dir or args.capture_root),
                    "--output-dir",
                    str(root),
                    "--_worker",
                    "resolve",
                    "--_result",
                    str(root / "checkpoint.json"),
                    "--revision",
                    args.revision,
                ]
                if args.offline:
                    command.append("--offline")
                try:
                    code, elapsed, _ = child(command, resolve_dir, args.timeout_seconds, env)
                except KeyboardInterrupt:
                    write_json(root / "summary.json", {"plan": plan, "status": "interrupted_during_checkpoint_resolution"})
                    zip_logs(root)
                    print(f"Interrupted; logs: {root / 'share-logs.zip'}")
                    return 130
                if code != 0:
                    summary = {
                        "plan": plan,
                        "status": "checkpoint_unavailable",
                        "resolve_exit_code": code,
                        "resolve_seconds": elapsed,
                    }
                    write_json(root / "summary.json", summary)
                    zip_logs(root)
                    print(f"Checkpoint unavailable; logs: {root / 'share-logs.zip'}")
                    return 1
                checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
                summary = {"plan": plan, "status": "incomplete", "checkpoint_commit": checkpoint["revision"], "cases": []}
                for case in plans:
                    count = case["views"]
                    directory = root / f"views-{count:02d}"
                    directory.mkdir()
                    manifest = Path(temp) / f"views-{count:02d}.json"
                    manifest.write_text(
                        json.dumps([str(images[index]) for index in case["source_indices"]]), encoding="utf-8"
                    )
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--images-dir",
                        str(args.images_dir or args.capture_root),
                        "--output-dir",
                        str(root),
                        "--_worker",
                        "infer",
                        "--_result",
                        str(directory / "result.json"),
                        "--_snapshot",
                        checkpoint["snapshot_path"],
                        "--_images_json",
                        str(manifest),
                    ]
                    if args.no_glb:
                        command.append("--no-glb")
                    # No inference-time network: only the explicit checkpoint resolution may download weights.
                    infer_env = {**env, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                    print(f"Running {MODEL_ID} with {count} browser views...", flush=True)
                    try:
                        code, elapsed, gpu_samples = child(
                            command, directory, args.timeout_seconds, infer_env, measure_gpu=True
                        )
                    except KeyboardInterrupt:
                        summary["cases"].append({"views": count, "status": "interrupted"})
                        summary["status"] = "interrupted"
                        write_json(root / "summary.json", summary)
                        zip_logs(root)
                        print(f"Interrupted; logs: {root / 'share-logs.zip'}")
                        return 130
                    result_file = directory / "result.json"
                    result = (
                        json.loads(result_file.read_text(encoding="utf-8"))
                        if result_file.exists()
                        else {"status": "timeout" if code == 124 else "process_failed"}
                    )
                    summary["cases"].append(
                        {"views": count, "exit_code": code, "wall_seconds_including_load": elapsed, **gpu_samples, **result}
                    )
                    write_json(root / "summary.json", summary)
                    if result["status"] in {"oom", "unavailable", "timeout", "process_failed", "failed", "preview_failed"}:
                        break
                summary["status"] = (
                    "completed"
                    if all(case["status"] == "model_completed" for case in summary["cases"])
                    else "partial_or_failed"
                )
                write_json(root / "summary.json", summary)
                zip_logs(root)
                print(f"Report: {root / 'summary.json'}")
                print(f"Shareable logs: {root / 'share-logs.zip'} (no input frames or GLB)")
                return 0 if any(case["status"] == "model_completed" for case in summary["cases"]) else 1
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"model smoke: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
