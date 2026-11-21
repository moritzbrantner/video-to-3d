"""Stdlib-only scoring rules shared by the TUM geometry ablation and its unit tests.

The TUM trajectory and Kinect depth are evaluation oracles. Nothing here feeds production
reconstruction; it only classifies already exported geometry against held-out measurements.
"""
from __future__ import annotations

import math

# A ray within this axial distance of the Kinect measurement counts as correct geometry.
TOLERANCE_M = 0.10
TIGHT_TOLERANCE_M = 0.05

RAY_CLASSES = ("correct", "false_positive_in_front", "behind_truth", "missing")


def classify_ray(truth_m: float, hit_m: float | None, tolerance: float = TOLERANCE_M) -> str:
    """Classify one held-out ray that has a valid Kinect measurement.

    `missing`: no reconstructed surface along the ray (lost geometry).
    `false_positive_in_front`: the first surface is in front of the measured one, so it is
    geometry where the sensor saw free space (a floater, misplaced layer or wrong depth).
    `behind_truth`: the ray passes through a hole in the true surface and hits something farther.
    """
    if not truth_m > 0:
        raise ValueError("classify_ray requires a valid measured depth")
    if hit_m is None:
        return "missing"
    error = hit_m - truth_m
    if abs(error) <= tolerance:
        return "correct"
    return "false_positive_in_front" if error < 0 else "behind_truth"


def ray_summary(counts: dict[str, int]) -> dict[str, float]:
    valid = sum(counts.get(name, 0) for name in RAY_CLASSES)
    if not valid:
        raise ValueError("no valid held-out rays")
    return {name: counts.get(name, 0) / valid for name in RAY_CLASSES}


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile matching numpy's default."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# Stage order of the production observed-surface route. Each stage can only lose coverage
# that the previous stage still had.
STAGES = (
    "camera_poses",
    "depth_estimation",
    "cross_view_verification",
    "meshing",
    "surface_assembly",
)


def attribute_loss(oracle: dict[str, float]) -> dict:
    """Attribute the held-out correct-ray fraction to the first stage that loses it.

    `oracle` holds the fraction of held-out valid rays reconstructed within tolerance by:
    - `reference_pose_measured_depth_tsdf`: oracle cameras and oracle depth, independent fusion
    - `estimated_pose_measured_depth_tsdf`: estimated cameras, oracle depth
    - `estimated_pose_estimated_depth_depthmaps`: estimated depth maps before any fusion
    - `estimated_pose_estimated_depth_verified`: after Rust cross-view verification
    - `estimated_pose_estimated_depth_rust`: exported production surface
    - `estimated_pose_estimated_depth_tsdf`: same estimated inputs through reference fusion

    Fusion-method and stage losses are reported separately so a weak surface assembly is not
    mistaken for weak depth.
    """
    required = (
        "reference_pose_measured_depth_tsdf",
        "estimated_pose_measured_depth_tsdf",
        "estimated_pose_estimated_depth_tsdf",
        "estimated_pose_estimated_depth_rust",
    )
    missing = [key for key in required if key not in oracle]
    if missing:
        raise ValueError(f"missing ablation results: {', '.join(missing)}")
    ceiling = oracle["reference_pose_measured_depth_tsdf"]
    pose = ceiling - oracle["estimated_pose_measured_depth_tsdf"]
    depth = oracle["estimated_pose_measured_depth_tsdf"] - oracle["estimated_pose_estimated_depth_tsdf"]
    assembly = oracle["estimated_pose_estimated_depth_tsdf"] - oracle["estimated_pose_estimated_depth_rust"]
    losses = {"camera_poses": pose, "depth_estimation": depth, "surface_assembly": assembly}
    largest = max(losses, key=lambda name: (losses[name], -STAGES.index(name)))
    earliest = next((name for name in STAGES if losses.get(name, 0) >= 0.05), None)
    return {
        "ceiling": ceiling,
        "losses": losses,
        "largest_loss_stage": largest,
        "earliest_material_loss_stage": earliest,
    }


def markdown_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """Render rows as a Markdown table. `columns` holds (key, header, format)."""
    head = "| " + " | ".join(header for _, header, _ in columns) + " |"
    rule = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(columns))) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            value = row.get(key)
            if value is None:
                cells.append("–")
            elif fmt == "%":
                cells.append(f"{100 * value:.1f}%")
            elif fmt == "cm":
                cells.append(f"{100 * value:.1f}")
            elif fmt == "int":
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])
