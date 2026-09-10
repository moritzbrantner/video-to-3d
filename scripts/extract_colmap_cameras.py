#!/usr/bin/env python3
"""Convert COLMAP images.txt poses into frame-indexed camera centers."""

from __future__ import annotations

import argparse
import math
import pathlib
import re

FRAME_RE = re.compile(r"frame_(\d+)\.")


def rotation_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> tuple[tuple[float, ...], ...]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("zero COLMAP quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )


def camera_center(rotation: tuple[tuple[float, ...], ...], translation: tuple[float, float, float]) -> tuple[float, float, float]:
    # COLMAP stores world-to-camera X_cam = R X_world + t, hence C = -R^T t.
    return tuple(-sum(rotation[row][column] * translation[row] for row in range(3)) for column in range(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images_txt")
    parser.add_argument("output")
    args = parser.parse_args()

    rows: list[tuple[int, tuple[float, float, float]]] = []
    for raw_line in pathlib.Path(args.images_txt).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 10:
            continue
        match = FRAME_RE.search(columns[9])
        if not match:
            continue
        qw, qx, qy, qz = map(float, columns[1:5])
        translation = tuple(map(float, columns[5:8]))
        rotation = rotation_from_quaternion(qw, qx, qy, qz)
        rows.append((int(match.group(1)), camera_center(rotation, translation)))

    rows.sort(key=lambda row: row[0])
    output = ["frame_index\tcx\tcy\tcz"]
    output.extend(f"{index}\t{center[0]:.9f}\t{center[1]:.9f}\t{center[2]:.9f}" for index, center in rows)
    pathlib.Path(args.output).write_text("\n".join(output) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
