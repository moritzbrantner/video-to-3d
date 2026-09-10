#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


def parse_line(path: Path, prefix: str) -> dict[str, str]:
    line = next((line for line in path.read_text().splitlines() if line.startswith(prefix)), None)
    if line is None:
        raise ValueError(f"missing {prefix} metrics in {path}")
    return dict(re.findall(r"([a-z_]+)=([^ ]+)", line))


def numeric(metrics: dict[str, str], key: str) -> float:
    try:
        return float(metrics[key])
    except KeyError as error:
        raise ValueError(f"missing {key} metric") from error
    except ValueError as error:
        raise ValueError(f"invalid {key} metric: {metrics[key]}") from error


def wall_time(path: Path) -> float:
    document = json.loads(path.read_text())
    metric = next(
        (metric for metric in document["metrics"] if metric["id"] == "process.wall_time"),
        None,
    )
    if metric is None:
        raise ValueError(f"missing process.wall_time in {path}")
    return float(metric["statistics"]["median"])


def evaluate(
    rust: dict[str, str],
    colmap: dict[str, str],
    *,
    case: str,
    expected_images: int,
    max_normalized_pose_rmse: float,
) -> list[str]:
    errors: list[str] = []
    if rust.get("case") != case:
        errors.append(f"Rust metrics reported case {rust.get('case')!r}; expected {case!r}")
    if colmap.get("case") != case:
        errors.append(f"COLMAP metrics reported case {colmap.get('case')!r}; expected {case!r}")

    colmap_registered = int(numeric(colmap, "registered_images"))
    rust_registered = int(numeric(rust, "registered_images"))
    minimum_colmap_registered = max(2, expected_images - 2)
    if colmap_registered < minimum_colmap_registered:
        errors.append(
            f"COLMAP reference registered only {colmap_registered}/{expected_images} images; expected at least {minimum_colmap_registered}"
        )
    required_rust_registered = max(4, colmap_registered - 2)
    if rust_registered < required_rust_registered:
        errors.append(
            f"Rust registered {rust_registered}/{expected_images} images; golden envelope requires at least {required_rust_registered} when COLMAP registers {colmap_registered}"
        )

    rust_points = int(numeric(rust, "points"))
    if rust_points < 40:
        errors.append(
            f"Rust sparse map has only {rust_points} points; expected at least 40 on the golden fixture"
        )

    rust_error = numeric(rust, "median_reprojection_error_pixels")
    colmap_error = numeric(colmap, "mean_reprojection_error_pixels")
    if not math.isfinite(rust_error):
        errors.append("Rust reconstruction did not report a finite reprojection error")
    else:
        max_error = max(3.0, colmap_error * 4.0)
        if rust_error > max_error:
            errors.append(
                f"Rust reprojection error {rust_error:.3f}px exceeds golden envelope {max_error:.3f}px (COLMAP {colmap_error:.3f}px)"
            )

    pose_rmse = numeric(rust, "normalized_pose_rmse")
    if not math.isfinite(pose_rmse):
        errors.append("Rust reconstruction did not report a finite normalized pose RMSE")
    elif pose_rmse > max_normalized_pose_rmse:
        errors.append(
            f"Rust normalized pose RMSE {pose_rmse:.3f} exceeds golden envelope {max_normalized_pose_rmse:.3f}"
        )

    return errors


def render_markdown(
    rust: dict[str, str],
    colmap: dict[str, str],
    *,
    case: str,
    expected_images: int,
    max_normalized_pose_rmse: float,
    rust_runtime: Path | None,
    colmap_runtime: Path | None,
    errors: list[str],
) -> str:
    rust_registered = int(numeric(rust, "registered_images"))
    colmap_registered = int(numeric(colmap, "registered_images"))
    rust_error = numeric(rust, "median_reprojection_error_pixels")
    colmap_error = numeric(colmap, "mean_reprojection_error_pixels")
    pose_rmse = numeric(rust, "normalized_pose_rmse")

    runtime_rows = ""
    if rust_runtime and colmap_runtime:
        rust_ms = wall_time(rust_runtime)
        colmap_ms = wall_time(colmap_runtime)
        ratio = rust_ms / colmap_ms if colmap_ms > 0 else math.inf
        runtime_rows = (
            f"\n| runtime-profiler median wall time | {rust_ms:.1f} ms | {colmap_ms:.1f} ms | {ratio:.2f}× COLMAP |\n"
        )

    return f"""#### COLMAP golden reference — {case}

| Evidence | video-to-3d | COLMAP | Gate |
| --- | ---: | ---: | --- |
| Registered images | {rust_registered}/{expected_images} | {colmap_registered}/{expected_images} | Rust ≥ max(4, COLMAP−2); COLMAP ≥ {max(2, expected_images - 2)} |
| Sparse points | {int(numeric(rust, 'points'))} | {int(numeric(colmap, 'points'))} | Rust ≥ 40; COLMAP count diagnostic |
| Reprojection error | {rust_error:.3f} px median | {colmap_error:.3f} px mean | Rust ≤ max(3 px, 4× COLMAP) |
| Normalized camera-center RMSE | {pose_rmse:.3f} | known trajectory | Rust ≤ {max_normalized_pose_rmse:.3f} after Sim(3) alignment |
{runtime_rows}
Status: {'PASS' if not errors else 'FAIL'}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--expected-images", type=int, default=8)
    parser.add_argument("--max-normalized-pose-rmse", type=float, default=0.25)
    parser.add_argument("--rust-runtime", type=Path)
    parser.add_argument("--colmap-runtime", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rust = parse_line(args.rust, "golden-rust")
        colmap = parse_line(args.colmap, "golden-colmap")
        errors = evaluate(
            rust,
            colmap,
            case=args.case,
            expected_images=args.expected_images,
            max_normalized_pose_rmse=args.max_normalized_pose_rmse,
        )
        markdown = render_markdown(
            rust,
            colmap,
            case=args.case,
            expected_images=args.expected_images,
            max_normalized_pose_rmse=args.max_normalized_pose_rmse,
            rust_runtime=args.rust_runtime,
            colmap_runtime=args.colmap_runtime,
            errors=errors,
        )
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"golden-gate: {error}")
        return 1

    print(markdown)
    if args.markdown:
        args.markdown.write_text(markdown)

    if errors:
        for error in errors:
            print(f"golden-gate: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
