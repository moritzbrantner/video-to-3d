#!/usr/bin/env python3
"""Prepare local/yt-dlp video frames and invoke the existing MapAnything smoke harness.

This is a local experimental hardware shortcut, not a browser-capture receipt
or the Rust-owned production reconstruction pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "smoke_mapanything.py"
DEFAULT_CHECKPOINT = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".ts"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(data: object) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def video_candidates(root: Path) -> list[Path]:
    return sorted(
        candidate for candidate in root.glob("source.*")
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS
    )


def invoke(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise ValueError(f"command exited {result.returncode}: {command[0]}")


def python_bin() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "video-to-3d"
    return Path(os.environ.get("MAPANYTHING_PYTHON") or
                (Path(os.environ.get("MAPANYTHING_VENV", str(cache / "mapanything-venv"))) / "bin/python"))


def download(url: str, root: Path, offline: bool) -> Path:
    cached = video_candidates(root)
    if len(cached) > 1:
        raise ValueError(f"ambiguous cached video files in {root}; choose a fresh output root")
    if cached:
        print(f"Reusing downloaded video: {cached[0]}")
        return cached[0]
    if offline:
        raise ValueError("--offline requires an already downloaded video in the output cache")
    py = python_bin()
    if py.is_file():
        command = [str(py), "-m", "yt_dlp"]
    elif shutil.which("yt-dlp"):
        command = ["yt-dlp"]
    else:
        raise ValueError("yt-dlp is unavailable; run ./scripts/setup_mapanything.sh first")
    invoke(command + ["--no-playlist", "-f", "bv*/b", "-o", str(root / "source.%(ext)s"), url])
    cached = video_candidates(root)
    if len(cached) != 1:
        raise ValueError("yt-dlp did not produce exactly one supported video file")
    return cached[0]


def validate_video(video: Path) -> None:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=False,
    )
    if probe.returncode or not probe.stdout.strip():
        raise ValueError(f"FFprobe cannot read a video stream: {video}")


def validate_frames(root: Path, expected: dict, min_frames: int) -> int:
    manifest_path = root / "frames.json"
    if not manifest_path.is_file():
        raise ValueError(f"incomplete frame cache at {root}; move it aside and retry")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid frame cache manifest: {root}") from error
    if manifest.get("input") != expected:
        raise ValueError(f"frame cache settings changed unexpectedly: {root}")
    entries = manifest.get("frames")
    if not isinstance(entries, list) or len(entries) < min_frames:
        raise ValueError(f"frame cache has fewer than {min_frames} frames: {root}")
    for index, digest in enumerate(entries):
        frame = root / f"frame-{index:04d}.png"
        if not frame.is_file() or sha256_file(frame) != digest:
            raise ValueError(f"frame cache failed integrity validation: {frame}")
    return len(entries)


def prepare_frames(video: Path, case_root: Path, args: argparse.Namespace, min_frames: int) -> Path:
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
    if version.returncode or not version.stdout:
        raise ValueError("FFmpeg is unavailable; install ffmpeg and ffprobe")
    expected = {
        "video_sha256": sha256_file(video), "fps": args.fps, "start_seconds": args.start_seconds,
        "max_frames": args.max_frames, "ffmpeg": version.stdout.splitlines()[0],
    }
    frame_dir = case_root / f"frames-{fingerprint(expected)[:16]}"
    if frame_dir.exists():
        count = validate_frames(frame_dir, expected, min_frames)
        print(f"Reusing {count} verified frames: {frame_dir}")
        return frame_dir
    with tempfile.TemporaryDirectory(prefix=".frames-", dir=case_root) as tmp:
        temporary = Path(tmp)
        invoke([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", str(args.start_seconds),
            "-i", str(video), "-vf", f"fps={args.fps:g}", "-frames:v", str(args.max_frames),
            "-start_number", "0", str(temporary / "frame-%04d.png"),
        ])
        frames = sorted(temporary.glob("frame-*.png"))
        if len(frames) < min_frames:
            raise ValueError(
                f"only {len(frames)} frames extracted; need {min_frames}; try --fps 2 or a longer clip"
            )
        manifest = {"input": expected, "frames": [sha256_file(path) for path in frames]}
        (temporary / "frames.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.rename(frame_dir)
    print(f"Extracted {len(frames)} lossless frames: {frame_dir}")
    return frame_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="one YouTube/video URL or a local video file")
    parser.add_argument("--output-root", type=Path, default=ROOT / "target/mapanything")
    parser.add_argument("--view-counts", default="8", help="comma-separated view counts, default 8")
    parser.add_argument("--fps", type=float, default=1.0, help="lossless frame extraction rate, default 1")
    parser.add_argument("--max-frames", type=int, default=16, help="maximum extracted frames, default 16")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="video time offset, default 0")
    parser.add_argument("--revision", default=DEFAULT_CHECKPOINT, help="immutable HF checkpoint commit")
    parser.add_argument("--offline", action="store_true", help="no video or model downloads")
    parser.add_argument("--prepare-only", action="store_true", help="prepare/check frames without CUDA inference")
    parser.add_argument("--dry-run", action="store_true", help="show settings without downloads or file changes")
    args = parser.parse_args()

    try:
        counts = [int(value) for value in args.view_counts.split(",")]
        if not counts or counts != sorted(set(counts)) or counts[0] < 2 or counts[-1] > args.max_frames:
            raise ValueError("--view-counts must be increasing integers from 2 through --max-frames")
        if not math.isfinite(args.fps) or args.fps <= 0 or args.fps > 60:
            raise ValueError("--fps must be a positive rate no greater than 60")
        if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
            raise ValueError("--start-seconds must be nonnegative")
        if args.max_frames > 256:
            raise ValueError("--max-frames must be at most 256")
        is_url = args.video.startswith(("https://", "http://"))
        if is_url:
            identity = fingerprint({"url": args.video})
        else:
            local = Path(args.video).expanduser().resolve(strict=True)
            if not local.is_file():
                raise ValueError(f"video is not a regular file: {local}")
            identity = sha256_file(local)
        case_root = args.output_root.expanduser().resolve() / f"{'url' if is_url else 'file'}-{identity[:16]}"
        if args.dry_run:
            print(f"Dry run: {args.video} -> {case_root}; views={args.view_counts}, fps={args.fps:g}, max_frames={args.max_frames}")
            return 0
        for required in ("ffmpeg", "ffprobe"):
            if not shutil.which(required):
                raise ValueError(f"{required} is missing; install FFmpeg first")
        case_root.mkdir(parents=True, exist_ok=True)
        video = download(args.video, case_root, args.offline) if is_url else local
        validate_video(video)
        frames = prepare_frames(video, case_root, args, counts[-1])
        if args.prepare_only:
            print(f"Frames ready for the smoke test: {frames}")
            return 0
        py = python_bin()
        if not py.is_file():
            if args.offline:
                raise ValueError("--offline requires the prepared MapAnything Python environment")
            print("Preparing the MapAnything environment...", flush=True)
            invoke(["bash", str(ROOT / "scripts/setup_mapanything.sh")])
        if not py.is_file():
            raise ValueError("MapAnything Python is missing; run ./scripts/setup_mapanything.sh")
        views = "-".join(map(str, counts))
        output = case_root / "runs" / frames.name / f"views-{views}-rev-{args.revision[:12]}"
        print(f"Running MapAnything: {frames} -> {output}", flush=True)
        command = [
            str(py), str(SMOKE), "--images-dir", str(frames), "--output-dir", str(output),
            "--numbered-runs", "--view-counts", args.view_counts, "--revision", args.revision,
        ]
        if args.offline:
            command.append("--offline")
        return subprocess.run(command, check=False).returncode
    except (ValueError, OSError) as error:
        parser.exit(2, f"MapAnything video: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
