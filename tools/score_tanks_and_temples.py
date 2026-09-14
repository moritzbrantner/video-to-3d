#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

MAX_GROUND_TRUTH_POINTS = 500_000
MAX_RECONSTRUCTION_POINTS = 250_000


def load_case(manifest_path: Path, case_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for case in manifest.get("cases", []):
        if case.get("id") == case_id:
            if case.get("tier") != "quantitative":
                raise RuntimeError(f"{case_id} is not a quantitative benchmark case")
            return case
    raise RuntimeError(f"unknown real-video benchmark case {case_id}")


def read_trajectory(path: Path) -> np.ndarray:
    centers: list[np.ndarray] = []
    with path.open(encoding="utf-8") as handle:
        while True:
            metadata = handle.readline()
            if not metadata:
                break
            rows = []
            for _ in range(4):
                row = handle.readline()
                if not row:
                    raise RuntimeError(f"truncated trajectory matrix in {path}")
                values = np.fromstring(row, dtype=np.float64, sep=" ")
                if values.size != 4:
                    raise RuntimeError(f"invalid trajectory row in {path}: {row.strip()!r}")
                rows.append(values)
            matrix = np.vstack(rows)
            centers.append(matrix[:3, 3])
    if not centers:
        raise RuntimeError(f"reference trajectory is empty: {path}")
    return np.vstack(centers)


def read_mapping(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        sampled_count = int(handle.readline().strip())
        _total_frames = int(handle.readline().strip())
        rows = []
        for _ in range(sampled_count):
            values = handle.readline().split()
            if len(values) != 2:
                raise RuntimeError(f"invalid mapping row in {path}")
            rows.append((int(values[0]), int(values[1])))
    return np.asarray(rows, dtype=np.int64)


def read_cameras(path: Path) -> list[dict[str, float | int]]:
    cameras: list[dict[str, float | int]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cameras.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                }
            )
    return cameras


def transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    transformed = homogeneous @ matrix.T
    return transformed[:, :3] / transformed[:, 3:4]


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3 or len(target) != len(source):
        raise RuntimeError("at least three paired camera centers are required for similarity alignment")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if np.linalg.matrix_rank(source_centered) < 2:
        raise RuntimeError("accepted camera centers are too close to collinear for a stable 3D similarity alignment")

    covariance = (target_centered.T @ source_centered) / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    variance = float(np.sum(source_centered * source_centered) / len(source))
    if variance <= 1.0e-12:
        raise RuntimeError("accepted camera centers have no usable trajectory scale")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("camera correspondence produced an invalid similarity scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def camera_correspondences(
    cameras: list[dict[str, float | int]],
    sample_times: list[float],
    frame_rate: float,
    mapping: np.ndarray,
    reference_centers: np.ndarray,
    ground_truth_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    candidates: dict[int, tuple[float, np.ndarray, np.ndarray, dict[str, float | int]]] = {}
    mapping_frame_times = (mapping[:, 1].astype(np.float64) - 1.0) / frame_rate

    for camera in cameras:
        frame_index = int(camera["frame_index"])
        if frame_index < 0 or frame_index >= len(sample_times):
            continue
        sample_time = float(sample_times[frame_index])
        mapping_row = int(np.argmin(np.abs(mapping_frame_times - sample_time)))
        reference_index = int(mapping[mapping_row, 0])
        if reference_index < 0 or reference_index >= len(reference_centers):
            continue
        time_delta = abs(float(mapping_frame_times[mapping_row]) - sample_time)
        source = np.asarray([camera["x"], camera["y"], camera["z"]], dtype=np.float64)
        target = apply_homogeneous(
            reference_centers[reference_index : reference_index + 1],
            ground_truth_transform,
        )[0]
        evidence = {
            "sample_frame_index": frame_index,
            "sample_time_seconds": sample_time,
            "reference_index": reference_index,
            "reference_video_frame": int(mapping[mapping_row, 1]),
            "reference_time_delta_seconds": time_delta,
        }
        previous = candidates.get(reference_index)
        if previous is None or time_delta < previous[0]:
            candidates[reference_index] = (time_delta, source, target, evidence)

    ordered = [candidates[index] for index in sorted(candidates)]
    if not ordered:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return (
        np.vstack([item[1] for item in ordered]),
        np.vstack([item[2] for item in ordered]),
        [item[3] for item in ordered],
    )


def read_ply_points(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    if len(vertex) == 0:
        return np.empty((0, 3), dtype=np.float64)
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64, copy=False)
    return points[np.all(np.isfinite(points), axis=1)]


def bounded_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def score_geometry(
    points: np.ndarray,
    ground_truth: np.ndarray,
    tau: float,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    if len(points) == 0:
        return {"status": "no-points", "points_total": 0, "points_scored": 0}

    transformed = transform_points(points, scale, rotation, translation)
    reconstruction = bounded_points(transformed, MAX_RECONSTRUCTION_POINTS)
    gt_sample = bounded_points(ground_truth, MAX_GROUND_TRUTH_POINTS)
    gt_tree = cKDTree(gt_sample)
    reconstruction_tree = cKDTree(reconstruction)
    reconstruction_to_gt = gt_tree.query(reconstruction, k=1, workers=-1)[0]
    gt_to_reconstruction = reconstruction_tree.query(gt_sample, k=1, workers=-1)[0]

    precision = float(np.mean(reconstruction_to_gt < tau))
    recall = float(np.mean(gt_to_reconstruction < tau))
    f_score = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "status": "scored",
        "points_total": int(len(points)),
        "points_scored": int(len(reconstruction)),
        "ground_truth_points_scored": int(len(gt_sample)),
        "precision_at_tau": precision,
        "recall_at_tau": recall,
        "f_score_at_tau": f_score,
        "accuracy_median_ground_truth_units": float(np.median(reconstruction_to_gt)),
        "completeness_median_ground_truth_units": float(np.median(gt_to_reconstruction)),
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(f"{json.dumps(result, indent=2)}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a real-video case against Tanks and Temples references")
    parser.add_argument("--case", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/real-video-corpus.json"))
    parser.add_argument("--asset-root", type=Path, default=Path(".benchmark-data"))
    parser.add_argument("--output-root", type=Path, default=Path("target/real-video"))
    args = parser.parse_args()

    case = load_case(args.manifest, args.case)
    scene = case["asset"]["scene"]
    evaluation = case["evaluation"]
    case_output = args.output_root / args.case
    result_path = case_output / "ground-truth-score.json"
    sampling = json.loads((case_output / "sampling.json").read_text(encoding="utf-8"))
    frame_rate = float(sampling["source_metadata"]["frame_rate"])
    sample_times = [float(value) for value in sampling["plan"]["times"]]

    evaluation_dir = args.asset_root / "tanks-and-temples" / "evaluation" / scene
    reference_centers = read_trajectory(evaluation_dir / f"{scene}_COLMAP_SfM.log")
    mapping = read_mapping(evaluation_dir / f"{scene}_mapping_reference.txt")
    ground_truth_transform = np.loadtxt(evaluation_dir / f"{scene}_trans.txt", dtype=np.float64)
    if ground_truth_transform.shape != (4, 4):
        raise RuntimeError(f"{scene}_trans.txt must contain one 4x4 transform")
    cameras = read_cameras(case_output / "cameras.csv")
    source_centers, target_centers, correspondence_evidence = camera_correspondences(
        cameras,
        sample_times,
        frame_rate,
        mapping,
        reference_centers,
        ground_truth_transform,
    )

    result: dict[str, Any] = {
        "schema_version": "video-to-3d/tanks-and-temples-score/v1",
        "case_id": args.case,
        "scene": scene,
        "scoring": evaluation["scoring"],
        "official_tanks_and_temples_score": False,
        "distance_tau": float(evaluation["distance_tau"]),
        "reference_camera_count": int(len(reference_centers)),
        "camera_correspondences": correspondence_evidence,
        "alignment": {
            "status": "insufficient-camera-correspondences",
            "correspondence_count": int(len(source_centers)),
        },
        "geometry": {},
        "notes": [
            "This evidence uses the provider laser scan and COLMAP camera reference, but it is not the official cropped/ICP-refined Tanks and Temples leaderboard score.",
            "Ground-truth and reconstruction point sets are deterministically bounded before nearest-neighbor scoring so scheduled evidence stays memory-bounded.",
        ],
    }

    if len(source_centers) < 3 or np.linalg.matrix_rank(source_centers - source_centers.mean(axis=0)) < 2:
        write_result(result_path, result)
        print(f"real-video-ground-truth case={args.case} status=insufficient-camera-correspondences")
        return 0

    scale, rotation, translation = estimate_similarity(source_centers, target_centers)
    aligned_cameras = transform_points(source_centers, scale, rotation, translation)
    camera_errors = np.linalg.norm(aligned_cameras - target_centers, axis=1)
    result["alignment"] = {
        "status": "scored",
        "correspondence_count": int(len(source_centers)),
        "camera_center_rmse_ground_truth_units": float(np.sqrt(np.mean(camera_errors * camera_errors))),
        "camera_center_median_error_ground_truth_units": float(np.median(camera_errors)),
        "max_reference_time_delta_seconds": float(
            max(float(item["reference_time_delta_seconds"]) for item in correspondence_evidence)
        ),
        "similarity_scale": scale,
        "similarity_rotation": rotation.tolist(),
        "similarity_translation": translation.tolist(),
    }

    ground_truth = read_ply_points(evaluation_dir / f"{scene}.ply")
    if len(ground_truth) == 0:
        raise RuntimeError(f"ground-truth point cloud is empty for {scene}")
    tau = float(evaluation["distance_tau"])
    result["ground_truth_points_total"] = int(len(ground_truth))
    result["geometry"] = {
        "sparse": score_geometry(
            read_ply_points(case_output / "sparse.ply"),
            ground_truth,
            tau,
            scale,
            rotation,
            translation,
        ),
        "dense": score_geometry(
            read_ply_points(case_output / "dense.ply"),
            ground_truth,
            tau,
            scale,
            rotation,
            translation,
        ),
    }
    write_result(result_path, result)
    print(
        f"real-video-ground-truth case={args.case} status=scored "
        f"camera-rmse={result['alignment']['camera_center_rmse_ground_truth_units']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
