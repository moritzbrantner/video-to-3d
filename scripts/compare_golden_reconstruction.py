#!/usr/bin/env python3
"""Compare our camera trajectory and COLMAP against deterministic ground truth."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys


def read_truth(path: pathlib.Path) -> dict[int, tuple[float, float, float]]:
    rows: dict[int, tuple[float, float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        columns = line.split("\t")
        rows[int(columns[0])] = tuple(map(float, columns[3:6]))
    return rows


def read_cameras(path: pathlib.Path) -> dict[int, tuple[float, float, float]]:
    rows: dict[int, tuple[float, float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        columns = line.split("\t")
        if len(columns) != 4:
            continue
        rows[int(columns[0])] = tuple(map(float, columns[1:4]))
    return rows


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def trajectory_metrics(
    truth: dict[int, tuple[float, float, float]],
    cameras: dict[int, tuple[float, float, float]],
) -> dict[str, float | int | None]:
    common = sorted(set(truth) & set(cameras))
    predicted_distances: list[float] = []
    truth_distances: list[float] = []
    for offset, left in enumerate(common):
        for right in common[offset + 1 :]:
            predicted_distances.append(distance(cameras[left], cameras[right]))
            truth_distances.append(distance(truth[left], truth[right]))

    numerator = sum(predicted * expected for predicted, expected in zip(predicted_distances, truth_distances))
    denominator = sum(predicted * predicted for predicted in predicted_distances)
    scale = numerator / denominator if denominator > 1.0e-12 else None
    truth_scale_values = [value for value in truth_distances if value > 1.0e-9]
    truth_scale = sum(truth_scale_values) / len(truth_scale_values) if truth_scale_values else 1.0
    rmse = None
    if scale is not None and predicted_distances:
        squared = [
            (scale * predicted - expected) ** 2
            for predicted, expected in zip(predicted_distances, truth_distances)
        ]
        rmse = math.sqrt(sum(squared) / len(squared)) / max(truth_scale, 1.0e-9)

    loop_ratio = None
    loop_error = None
    first = min(truth)
    last = max(truth)
    if first in cameras and last in cameras:
        predicted_path = 0.0
        truth_path = 0.0
        for left, right in zip(common, common[1:]):
            predicted_path += distance(cameras[left], cameras[right])
            truth_path += distance(truth[left], truth[right])
        if predicted_path > 1.0e-9 and truth_path > 1.0e-9:
            loop_ratio = distance(cameras[first], cameras[last]) / predicted_path
            truth_loop_ratio = distance(truth[first], truth[last]) / truth_path
            loop_error = abs(loop_ratio - truth_loop_ratio)

    return {
        "registered_cameras": len(common),
        "total_cameras": len(truth),
        "coverage": len(common) / len(truth),
        "pairwise_distance_rmse": rmse,
        "loop_return_ratio": loop_ratio,
        "loop_return_error": loop_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--colmap", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    truth = read_truth(pathlib.Path(args.truth))
    ours = trajectory_metrics(truth, read_cameras(pathlib.Path(args.ours)))
    colmap = trajectory_metrics(truth, read_cameras(pathlib.Path(args.colmap)))
    report = {"ours": ours, "colmap": colmap}
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if not args.gate:
        return

    failures: list[str] = []
    total = len(truth)
    colmap_registered = int(colmap["registered_cameras"])
    ours_registered = int(ours["registered_cameras"])
    if colmap_registered < max(6, total - 2):
        failures.append(
            f"COLMAP reference registered only {colmap_registered}/{total} cameras; golden reference is unstable"
        )
    if ours_registered < max(4, colmap_registered - 2):
        failures.append(
            f"our reconstruction registered {ours_registered}/{total} cameras versus COLMAP {colmap_registered}/{total}"
        )

    ours_rmse = ours["pairwise_distance_rmse"]
    colmap_rmse = colmap["pairwise_distance_rmse"]
    if ours_rmse is None:
        failures.append("our reconstruction has insufficient camera geometry for pairwise-distance scoring")
    elif colmap_rmse is not None:
        allowed_rmse = max(0.22, float(colmap_rmse) * 3.0 + 0.08)
        if float(ours_rmse) > allowed_rmse:
            failures.append(
                f"our scale-invariant trajectory RMSE {ours_rmse:.4f} exceeds golden allowance {allowed_rmse:.4f} (COLMAP {colmap_rmse:.4f})"
            )

    ours_loop = ours["loop_return_error"]
    colmap_loop = colmap["loop_return_error"]
    if ours_loop is None:
        failures.append("our reconstruction did not preserve both ends of the return trajectory")
    elif colmap_loop is not None:
        allowed_loop = max(0.12, float(colmap_loop) + 0.08)
        if float(ours_loop) > allowed_loop:
            failures.append(
                f"our normalized loop-return error {ours_loop:.4f} exceeds golden allowance {allowed_loop:.4f} (COLMAP {colmap_loop:.4f})"
            )

    if failures:
        for failure in failures:
            print(f"GOLDEN FAILURE: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
