#!/usr/bin/env python3
"""Sample the committed Trevi clip using the browser reconstruction cadence."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess


def probe(video: pathlib.Path) -> tuple[float, int, int]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration:format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("Trevi fixture has no video stream")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    raw_duration = payload.get("format", {}).get("duration") or stream.get("duration")
    if raw_duration is None:
        raise RuntimeError("Trevi fixture has no usable duration")
    duration = float(raw_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Trevi fixture has invalid duration: {duration}")
    return duration, width, height


def javascript_round(value: float) -> int:
    return math.floor(value + 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    duration, source_width, source_height = probe(args.video)
    frames_per_second = 1.25
    max_frames = 18
    target_count = max(4, math.ceil(duration * frames_per_second))
    sample_count = min(max_frames, target_count)
    analysis_width = min(360, source_width)
    analysis_height = max(1, javascript_round(analysis_width / source_width * source_height))

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    times: list[float] = []
    for index in range(sample_count):
        fraction = 0.5 if sample_count == 1 else 0.05 + 0.9 * index / (sample_count - 1)
        time = min(duration - 0.001, max(0.0, duration * fraction))
        times.append(time)
        frame_path = args.output / f"frame-{index:02d}.ppm"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{time:.6f}",
                "-i",
                str(args.video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={analysis_width}:{analysis_height}:flags=bicubic",
                "-pix_fmt",
                "rgb24",
                "-f",
                "image2",
                "-vcodec",
                "ppm",
                str(frame_path),
            ],
            check=True,
        )

    sampling = {
        "source": str(args.video),
        "duration_seconds": duration,
        "source_width": source_width,
        "source_height": source_height,
        "frames_per_second": frames_per_second,
        "max_frames": max_frames,
        "sample_count": sample_count,
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "sample_times_seconds": times,
    }
    (args.output / "sampling.json").write_text(
        json.dumps(sampling, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sampling, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
