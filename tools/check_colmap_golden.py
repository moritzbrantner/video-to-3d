#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


def parse_line(path: Path, prefix: str) -> dict[str, float]:
    line = next((line for line in path.read_text().splitlines() if line.startswith(prefix)), None)
    if line is None:
        raise SystemExit(f"missing {prefix} metrics in {path}")
    result: dict[str, float] = {}
    for key, value in re.findall(r"([a-z_]+)=([^ ]+)", line):
        result[key] = float(value)
    return result


def wall_time(path: Path) -> float:
    document = json.loads(path.read_text())
    metric = next((metric for metric in document["metrics"] if metric["id"] == "process.wall_time"), None)
    if metric is None:
        raise SystemExit(f"missing process.wall_time in {path}")
    return float(metric["statistics"]["median"])


parser = argparse.ArgumentParser()
parser.add_argument("--rust", type=Path, required=True)
parser.add_argument("--colmap", type=Path, required=True)
parser.add_argument("--rust-runtime", type=Path)
parser.add_argument("--colmap-runtime", type=Path)
parser.add_argument("--markdown", type=Path)
args = parser.parse_args()

rust = parse_line(args.rust, "golden-rust")
colmap = parse_line(args.colmap, "golden-colmap")

errors: list[str] = []
colmap_registered = int(colmap["registered_images"])
rust_registered = int(rust["registered_images"])
if colmap_registered < 6:
    errors.append(f"COLMAP reference registered only {colmap_registered}/8 images")
required_rust_registered = max(4, colmap_registered - 2)
if rust_registered < required_rust_registered:
    errors.append(
        f"Rust registered {rust_registered}/8 images; golden envelope requires at least {required_rust_registered} when COLMAP registers {colmap_registered}"
    )
if int(rust["points"]) < 40:
    errors.append(f"Rust sparse map has only {int(rust['points'])} points; expected at least 40 on the golden fixture")

rust_error = rust["median_reprojection_error_pixels"]
colmap_error = colmap["mean_reprojection_error_pixels"]
if not math.isfinite(rust_error):
    errors.append("Rust reconstruction did not report a finite reprojection error")
else:
    max_error = max(3.0, colmap_error * 4.0)
    if rust_error > max_error:
        errors.append(
            f"Rust reprojection error {rust_error:.3f}px exceeds golden envelope {max_error:.3f}px (COLMAP {colmap_error:.3f}px)"
        )

runtime_rows = ""
if args.rust_runtime and args.colmap_runtime:
    rust_ms = wall_time(args.rust_runtime)
    colmap_ms = wall_time(args.colmap_runtime)
    ratio = rust_ms / colmap_ms if colmap_ms > 0 else math.inf
    runtime_rows = (
        f"\n| runtime-profiler median wall time | {rust_ms:.1f} ms | {colmap_ms:.1f} ms | {ratio:.2f}× COLMAP |\n"
    )

markdown = f"""### COLMAP golden reference

| Evidence | video-to-3d | COLMAP | Gate |
| --- | ---: | ---: | --- |
| Registered images | {rust_registered}/8 | {colmap_registered}/8 | Rust ≥ max(4, COLMAP−2) |
| Sparse points | {int(rust['points'])} | {int(colmap['points'])} | Rust ≥ 40; COLMAP count diagnostic |
| Reprojection error | {rust_error:.3f} px median | {colmap_error:.3f} px mean | Rust ≤ max(3 px, 4× COLMAP) |
{runtime_rows}
Status: {'PASS' if not errors else 'FAIL'}
"""
print(markdown)
if args.markdown:
    args.markdown.write_text(markdown)

if errors:
    for error in errors:
        print(f"golden-gate: {error}")
    raise SystemExit(1)
