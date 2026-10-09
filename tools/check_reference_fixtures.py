#!/usr/bin/env python3
"""Check a reference fixture (issue #128) against its target and recorded baseline.

Each fixture in `benchmarks/reference-fixtures.json` states the outcome its exact truth
demands (`target`) and the checks the current pipeline is known to fail
(`baseline.expected_failures`, each tied to the issue that will fix it). The gate:

- PASS: every target check passes and none is expected to fail.
- EXPECTED FAILURE: exactly the recorded checks fail. Reported, not fatal.
- UNEXPECTED PASS: a recorded failure now passes. Fatal until the baseline is updated,
  so the record never claims a failure that no longer exists.
- FAIL: any other check fails. Fatal.

COLMAP is a reference implementation, not ground truth: its result is reported next to
the fixture's `colmap_expectation` but never gates.
"""

import argparse
import json
import math
import re
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).parents[1] / "benchmarks" / "reference-fixtures.json"

TARGET_CHECKS = {
    "min_registered_images": ("registered_images", "≥"),
    "max_registered_images": ("registered_images", "≤"),
    "max_normalized_pose_rmse": ("normalized_pose_rmse", "≤"),
    "decisive_gate": ("decisive_gate", "="),
    "mesh_components": ("mesh_components", "="),
    "min_cross_reference_triangles": ("cross_reference_triangles", "≥"),
}

REPORTED_METRICS = [
    "registered_images",
    "points",
    "median_reprojection_error_pixels",
    "normalized_pose_rmse",
    "decisive_gate",
    "dense_points",
    "mesh_triangles",
    "mesh_components",
    "per_patch_components",
    "largest_component_share",
    "meshed_reference_patches",
    "cross_reference_triangles",
]


def parse_line(path: Path, prefix: str) -> dict[str, str]:
    line = next((line for line in path.read_text().splitlines() if line.startswith(prefix)), None)
    if line is None:
        raise ValueError(f"missing {prefix} metrics in {path}")
    return dict(re.findall(r"([a-z_]+)=([^ ]+)", line))


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported reference fixture manifest schema in {path}")
    seen = set()
    for fixture in manifest["fixtures"]:
        identifier = fixture["id"]
        if identifier in seen:
            raise ValueError(f"duplicate reference fixture {identifier!r}")
        seen.add(identifier)
        unknown = set(fixture["target"]) - set(TARGET_CHECKS)
        if unknown:
            raise ValueError(f"{identifier}: unknown target checks {sorted(unknown)}")
        for failure in fixture["baseline"]["expected_failures"]:
            if failure["check"] not in fixture["target"]:
                raise ValueError(
                    f"{identifier}: expected failure {failure['check']!r} is not a target check"
                )
            if not isinstance(failure.get("issue"), int):
                raise ValueError(f"{identifier}: expected failure {failure['check']!r} needs an issue")
    return manifest


def fixture_by_id(manifest: dict, case: str) -> dict:
    fixture = next((fixture for fixture in manifest["fixtures"] if fixture["id"] == case), None)
    if fixture is None:
        raise ValueError(f"unknown reference fixture {case!r}")
    return fixture


def number(metrics: dict[str, str], key: str) -> float:
    try:
        return float(metrics[key])
    except KeyError as error:
        raise ValueError(f"missing {key} metric") from error
    except ValueError as error:
        raise ValueError(f"invalid {key} metric: {metrics[key]}") from error


def check_passes(check: str, expected, rust: dict[str, str]) -> bool:
    metric, _ = TARGET_CHECKS[check]
    if check == "decisive_gate":
        if metric not in rust:
            raise ValueError(f"missing {metric} metric")
        return rust[metric] == expected
    value = number(rust, metric)
    if not math.isfinite(value):
        return False
    if check.startswith("min_"):
        return value >= expected
    if check.startswith("max_"):
        return value <= expected
    return value == expected


def evaluate(fixture: dict, rust: dict[str, str]) -> tuple[str, list[str]]:
    """Return the status and the failure messages that make it fatal, if any."""
    if rust.get("case") != fixture["id"]:
        return "FAIL", [f"Rust metrics reported case {rust.get('case')!r}; expected {fixture['id']!r}"]
    failing = {
        check for check, expected in fixture["target"].items() if not check_passes(check, expected, rust)
    }
    expected_failures = {failure["check"]: failure for failure in fixture["baseline"]["expected_failures"]}

    unexpected = sorted(failing - set(expected_failures))
    recovered = sorted(set(expected_failures) - failing)
    errors = []
    for check in unexpected:
        metric, relation = TARGET_CHECKS[check]
        errors.append(
            f"{fixture['id']}: {check} failed: {metric}={rust.get(metric)} (target {relation} {fixture['target'][check]})"
        )
    for check in recovered:
        errors.append(
            f"{fixture['id']}: expected failure {check} (#{expected_failures[check]['issue']}) now passes; "
            "remove it from benchmarks/reference-fixtures.json and record the new baseline"
        )
    if unexpected:
        return "FAIL", errors
    if recovered:
        return "UNEXPECTED PASS", errors
    if failing:
        return "EXPECTED FAILURE", []
    return "PASS", []


def colmap_summary(colmap: dict[str, str] | None) -> str:
    if colmap is None:
        return "no evidence"
    registered = colmap.get("registered_images", "?")
    if registered == "0":
        return "no model"
    return f"{registered} registered, {colmap.get('points', '?')} points, {colmap.get('mean_reprojection_error_pixels', '?')} px mean reprojection"


def render_markdown(fixture: dict, rust: dict[str, str], colmap: dict[str, str] | None, status: str, errors: list[str]) -> str:
    baseline = fixture["baseline"]["rust"]
    target_rows = []
    for check, expected in fixture["target"].items():
        metric, relation = TARGET_CHECKS[check]
        target_rows.append(f"{metric} {relation} {expected}")
    rows = "\n".join(
        f"| {metric} | {rust.get(metric, '—')} | {baseline.get(metric, '—')} |" for metric in REPORTED_METRICS
    )
    expected = "\n".join(
        f"- Expected failure `{failure['check']}` (#{failure['issue']}): {failure['reason']}"
        for failure in fixture["baseline"]["expected_failures"]
    )
    details = "\n".join(f"- {error}" for error in errors)
    notes = fixture["baseline"].get("notes", "")
    return f"""#### Reference fixture — {fixture['id']}

{fixture['intent']}

Target: {'; '.join(target_rows)}

| Evidence | video-to-3d | Baseline ({fixture['baseline']['recorded']}) |
| --- | ---: | ---: |
{rows}

COLMAP reference: {colmap_summary(colmap)} (expected: {fixture['colmap_expectation']}; reported, not gated)

{notes}

Status: {status}
{expected}
{details}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--list-cases", action="store_true", help="print fixture ids and exit")
    parser.add_argument("--case")
    parser.add_argument("--rust", type=Path)
    parser.add_argument("--colmap", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.list_cases:
        print("\n".join(fixture["id"] for fixture in manifest["fixtures"]))
        return 0
    if not args.case or not args.rust:
        raise SystemExit("--case and --rust are required")

    try:
        fixture = fixture_by_id(manifest, args.case)
        rust = parse_line(args.rust, "golden-rust")
        colmap = None
        if args.colmap:
            try:
                colmap = parse_line(args.colmap, "golden-colmap")
            except (OSError, ValueError):
                colmap = None
        status, errors = evaluate(fixture, rust)
        markdown = render_markdown(fixture, rust, colmap, status, errors)
    except (OSError, ValueError, KeyError) as error:
        status, errors = "FAIL", [f"incomplete reference fixture evidence: {error}"]
        markdown = f"#### Reference fixture — {args.case}\n\nStatus: FAIL\n\n- {errors[0]}\n"

    print(markdown)
    if args.markdown:
        args.markdown.write_text(markdown)
    for error in errors:
        print(f"reference-fixture-gate: {error}")
    return 1 if status in {"FAIL", "UNEXPECTED PASS"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
