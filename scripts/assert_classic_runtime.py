#!/usr/bin/env python3
"""Apply a deliberately broad Moonlight policy to runtime-profiler COLMAP evidence."""

from __future__ import annotations

import json
import os
import pathlib
import sys

MIN_RUNTIME_SCORE = 80


def main() -> None:
    evidence_path = os.environ.get("VIDEO_TO_3D_CLASSIC_SCORE_JSON")
    if not evidence_path:
        print("VIDEO_TO_3D_CLASSIC_SCORE_JSON is required", file=sys.stderr)
        raise SystemExit(2)

    report = json.loads(pathlib.Path(evidence_path).read_text(encoding="utf-8"))
    score = int(report["score"])
    metrics = {metric["id"]: metric for metric in report.get("metrics", [])}
    wall_time = metrics.get("process.wall_time")
    success = metrics.get("process.success_rate")

    if success:
        mean = next((item for item in success["statistics"] if item["statistic"] == "mean"), None)
        if mean and float(mean["candidate"]) < 1.0:
            print("our classic-reference workload did not succeed on every measured run", file=sys.stderr)
            raise SystemExit(1)

    if score < MIN_RUNTIME_SCORE:
        print(
            f"runtime-profiler score {score} is below the broad COLMAP-relative Moonlight floor {MIN_RUNTIME_SCORE}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if wall_time:
        median = next((item for item in wall_time["statistics"] if item["statistic"] == "median"), None)
        if median and float(median["reference"]) > 0.0:
            ratio = float(median["candidate"]) / float(median["reference"])
            print(
                f"classic runtime accepted: score={score}, ours/COLMAP median wall-time ratio={ratio:.3f}"
            )
            return
    print(f"classic runtime accepted: score={score}")


if __name__ == "__main__":
    main()
