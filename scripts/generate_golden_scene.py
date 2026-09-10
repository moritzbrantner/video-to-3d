#!/usr/bin/env python3
"""Generate one deterministic sparse-SfM scene for our Rust path and COLMAP."""

from __future__ import annotations

import argparse
import binascii
import math
import pathlib
import struct
import zlib

WIDTH = 320
HEIGHT = 240
FOCAL = 260.0
CAMERA_CENTERS = [
    (0.00, 0.00, 0.00),
    (0.12, 0.01, 0.00),
    (0.24, 0.02, 0.00),
    (0.36, 0.03, 0.00),
    (0.48, 0.02, 0.00),
    (0.36, 0.01, 0.00),
    (0.18, 0.00, 0.00),
    (0.02, 0.00, 0.00),
]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", binascii.crc32(payload) & 0xFFFFFFFF)


def write_rgba_png(path: pathlib.Path, rgba: bytearray) -> None:
    rows = bytearray()
    stride = WIDTH * 4
    for y in range(HEIGHT):
        rows.append(0)
        start = y * stride
        rows.extend(rgba[start : start + stride])
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(png_chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 6, 0, 0, 0)))
    png.extend(png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)))
    png.extend(png_chunk(b"IEND", b""))
    path.write_bytes(png)


def scene_points() -> list[tuple[float, float, float, int]]:
    points = []
    point_id = 0
    for gy in range(7):
        for gx in range(9):
            x = -1.50 + gx * 0.375
            y = -1.05 + gy * 0.35
            z = 3.8 + ((gx * 17 + gy * 13) % 6) * 0.52
            points.append((x, y, z, point_id))
            point_id += 1
    return points


def marker_value(point_id: int, dx: int, dy: int) -> int:
    if abs(dx) == 4 or abs(dy) == 4:
        return 238 if (point_id + dx - dy) % 2 == 0 else 38
    bit = (point_id * 37 + (dx + 4) * 11 + (dy + 4) * 19) & 7
    return 220 if bit in (0, 1, 3, 6) else 55


def render(center: tuple[float, float, float]) -> bytearray:
    rgba = bytearray([26, 26, 26, 255] * (WIDTH * HEIGHT))
    cx, cy, cz = center
    projected = []
    for x, y, z, point_id in scene_points():
        camera_z = z - cz
        if camera_z <= 0.1:
            continue
        px = int(round((x - cx) / camera_z * FOCAL + WIDTH * 0.5))
        py = int(round((y - cy) / camera_z * FOCAL + HEIGHT * 0.5))
        if px < 6 or px >= WIDTH - 6 or py < 6 or py >= HEIGHT - 6:
            continue
        projected.append((camera_z, px, py, point_id))

    # Farther marks first so nearer points deterministically win any overlap.
    projected.sort(reverse=True)
    for _, px, py, point_id in projected:
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                value = marker_value(point_id, dx, dy)
                offset = ((py + dy) * WIDTH + px + dx) * 4
                rgba[offset] = value
                rgba[offset + 1] = max(0, value - (point_id % 5) * 4)
                rgba[offset + 2] = min(255, value // 2 + (point_id % 7) * 8)
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="target/golden-scene")
    args = parser.parse_args()
    root = pathlib.Path(args.output).resolve()
    images = root / "images"
    raw = root / "raw"
    images.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    manifest = ["frame_index\timage\traw\tcx\tcy\tcz\tfocal\twidth\theight"]
    for frame_index, center in enumerate(CAMERA_CENTERS):
        name = f"frame_{frame_index:03d}"
        rgba = render(center)
        write_rgba_png(images / f"{name}.png", rgba)
        (raw / f"{name}.rgba").write_bytes(rgba)
        manifest.append(
            f"{frame_index}\timages/{name}.png\traw/{name}.rgba\t"
            f"{center[0]:.6f}\t{center[1]:.6f}\t{center[2]:.6f}\t"
            f"{FOCAL:.6f}\t{WIDTH}\t{HEIGHT}"
        )

    (root / "ground_truth.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
