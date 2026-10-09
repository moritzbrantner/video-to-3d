#!/usr/bin/env python3
"""Check a reference fixture stage by stage against its exact truth (issues #128, #129).

`benchmarks/reference-fixtures.json` lists the pipeline stages in order, each with
explicit tolerances, and per fixture the checks its truth demands plus the checks the
current pipeline is known to fail (each tied to the issue that will fix it). The
fixture's measurements come from `stage-report.json` (written by the
`reference_fixture` example) and `sampling-report.json` (written by
`tools/check_reference_sampling.ts`).

A stage that could not run (for example dense depth without registered cameras) fails
every one of its checks as "not reached". The report names the first failing stage.

- PASS: every check passes.
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
STAGE_REPORT_SCHEMA = "video-to-3d/reference-stage-report/v1"
BOUNDS = {"min": "≥", "max": "≤", "equals": "="}


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError(f"unsupported reference fixture manifest schema in {path}")
    order = manifest["stage_order"]
    if sorted(order) != sorted(manifest["stages"]):
        raise ValueError("stage_order and stages must name the same stages")
    for stage, definition in manifest["stages"].items():
        validate_checks(f"stage {stage}", definition["default_checks"])
    seen = set()
    for fixture in manifest["fixtures"]:
        identifier = fixture["id"]
        if identifier in seen:
            raise ValueError(f"duplicate reference fixture {identifier!r}")
        seen.add(identifier)
        checks = fixture_checks(manifest, fixture)
        for stage in fixture["checks"]:
            if stage not in manifest["stages"]:
                raise ValueError(f"{identifier}: unknown stage {stage!r}")
        for failure in fixture["baseline"]["expected_failures"]:
            stage = failure.get("stage")
            if stage not in checks:
                raise ValueError(f"{identifier}: expected failure names unchecked stage {stage!r}")
            if "check" in failure and failure["check"] not in checks[stage]:
                raise ValueError(
                    f"{identifier}: expected failure {stage}.{failure['check']} is not a check of that stage"
                )
            if not isinstance(failure.get("issue"), int):
                raise ValueError(f"{identifier}: expected failure in {stage} needs an issue")
    return manifest


def validate_checks(owner: str, checks: dict) -> None:
    for metric, bound in checks.items():
        if not isinstance(bound, dict) or len(bound) != 1 or next(iter(bound)) not in BOUNDS:
            raise ValueError(f"{owner}: check {metric!r} needs exactly one of {sorted(BOUNDS)}")


def fixture_checks(manifest: dict, fixture: dict) -> dict[str, dict]:
    """The fixture's checks per stage, in pipeline order."""
    checks = {}
    for stage in manifest["stage_order"]:
        if stage not in fixture["checks"]:
            continue
        stage_checks = fixture["checks"][stage]
        if stage_checks == "default":
            stage_checks = manifest["stages"][stage]["default_checks"]
        validate_checks(f"{fixture['id']} {stage}", stage_checks)
        checks[stage] = stage_checks
    return checks


def fixture_by_id(manifest: dict, case: str) -> dict:
    fixture = next((fixture for fixture in manifest["fixtures"] if fixture["id"] == case), None)
    if fixture is None:
        raise ValueError(f"unknown reference fixture {case!r}")
    return fixture


def load_stages(fixture_dir: Path, case: str) -> dict:
    report = json.loads((fixture_dir / "stage-report.json").read_text())
    if report.get("schema") != STAGE_REPORT_SCHEMA:
        raise ValueError(f"unsupported stage report schema {report.get('schema')!r}")
    if report.get("case") != case:
        raise ValueError(f"stage report is for case {report.get('case')!r}; expected {case!r}")
    stages = dict(report["stages"])
    sampling = fixture_dir / "sampling-report.json"
    stages["sampling"] = (
        json.loads(sampling.read_text())
        if sampling.exists()
        else {"available": False, "blocked_by": "sampling-report.json is missing"}
    )
    return stages


def check_result(bound: dict, value) -> bool:
    kind, expected = next(iter(bound.items()))
    if kind == "equals":
        return value == expected
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return value >= expected if kind == "min" else value <= expected


def evaluate(manifest: dict, fixture: dict, stages: dict) -> dict:
    """Per-stage check results, the first failing stage and the overall status."""
    checks = fixture_checks(manifest, fixture)
    expected = fixture["baseline"]["expected_failures"]

    def expectation(stage: str, metric: str):
        return next(
            (
                failure
                for failure in expected
                if failure["stage"] == stage and failure.get("check", metric) == metric
            ),
            None,
        )

    results = []
    for stage, stage_checks in checks.items():
        evidence = stages.get(stage) or {"available": False, "blocked_by": "no evidence"}
        available = evidence.get("available", False)
        metrics = evidence.get("metrics", {})
        rows = []
        for metric, bound in stage_checks.items():
            value = metrics.get(metric) if available else None
            passed = available and check_result(bound, value)
            rows.append(
                {
                    "metric": metric,
                    "bound": bound,
                    "value": value,
                    "passed": passed,
                    "reached": available,
                    "expected_failure": expectation(stage, metric),
                }
            )
        results.append(
            {
                "stage": stage,
                "available": available,
                "blocked_by": evidence.get("blocked_by"),
                "metrics": metrics,
                "checks": rows,
            }
        )

    errors = []
    for result in results:
        for row in result["checks"]:
            if not row["passed"] and row["expected_failure"] is None:
                actual = row["value"] if row["reached"] else f"not reached ({result['blocked_by']})"
                kind, target = next(iter(row["bound"].items()))
                errors.append(
                    f"{fixture['id']}: {result['stage']}.{row['metric']} failed: {actual} (target {BOUNDS[kind]} {target})"
                )
    unexpected = bool(errors)
    for failure in expected:
        covered = [
            row
            for result in results
            if result["stage"] == failure["stage"]
            for row in result["checks"]
            if failure.get("check", row["metric"]) == row["metric"]
        ]
        if all(row["passed"] for row in covered):
            name = failure["stage"] + (f".{failure['check']}" if "check" in failure else "")
            errors.append(
                f"{fixture['id']}: expected failure {name} (#{failure['issue']}) now passes; "
                "remove it from benchmarks/reference-fixtures.json and record the new baseline"
            )

    failing = [
        result["stage"]
        for result in results
        if any(not row["passed"] for row in result["checks"])
    ]
    first = failing[0] if failing else None
    if unexpected:
        status = "FAIL"
    elif errors:
        status = "UNEXPECTED PASS"
    elif failing:
        status = "EXPECTED FAILURE"
    else:
        status = "PASS"
    return {"status": status, "errors": errors, "first_failing_stage": first, "stages": results}


def format_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).lower() if isinstance(value, bool) else str(value)


def colmap_summary(colmap: dict[str, str] | None) -> str:
    if colmap is None:
        return "no evidence"
    registered = colmap.get("registered_images", "?")
    if registered == "0":
        return "no model"
    return f"{registered} registered, {colmap.get('points', '?')} points, {colmap.get('mean_reprojection_error_pixels', '?')} px mean reprojection"


def parse_line(path: Path, prefix: str) -> dict[str, str]:
    line = next((line for line in path.read_text().splitlines() if line.startswith(prefix)), None)
    if line is None:
        raise ValueError(f"missing {prefix} metrics in {path}")
    return dict(re.findall(r"([a-z_]+)=([^ ]+)", line))


def render_markdown(fixture: dict, evaluation: dict, colmap: dict[str, str] | None) -> str:
    rows = []
    for result in evaluation["stages"]:
        for row in result["checks"]:
            kind, target = next(iter(row["bound"].items()))
            if row["passed"]:
                outcome = "pass"
            elif row["expected_failure"] is not None:
                outcome = f"expected failure (#{row['expected_failure']['issue']})"
            else:
                outcome = "**FAIL**"
            value = format_value(row["value"]) if row["reached"] else "not reached"
            rows.append(
                f"| {result['stage']} | {row['metric']} | {value} | {BOUNDS[kind]} {format_value(target)} | {outcome} |"
            )
    first = evaluation["first_failing_stage"]
    recorded = fixture["baseline"].get("first_failing_stage")
    if first is None:
        first_line = "First failing stage: none"
    else:
        first_line = f"First failing stage: **{first}**"
    if first != recorded:
        first_line += f" (baseline: {recorded or 'none'})"
    expected = "\n".join(
        f"- Expected failure `{failure['stage']}{'.' + failure['check'] if 'check' in failure else ''}` "
        f"(#{failure['issue']}): {failure['reason']}"
        for failure in fixture["baseline"]["expected_failures"]
    )
    details = "\n".join(f"- {error}" for error in evaluation["errors"])
    notes = fixture["baseline"].get("notes", "")
    table = "\n".join(rows)
    return f"""#### Reference fixture — {fixture['id']}

{fixture['intent']}

Status: **{evaluation['status']}**. {first_line}.

| Stage | Check | video-to-3d | Tolerance | Result |
| --- | --- | ---: | ---: | --- |
{table}

COLMAP reference: {colmap_summary(colmap)} (expected: {fixture['colmap_expectation']}; reported, not gated)

{notes}

{expected}
{details}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--list-cases", action="store_true", help="print fixture ids and exit")
    parser.add_argument("--case")
    parser.add_argument("--fixture-dir", type=Path, help="directory with stage-report.json and sampling-report.json")
    parser.add_argument("--colmap", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path, help="write the machine-readable per-stage result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.list_cases:
        print("\n".join(fixture["id"] for fixture in manifest["fixtures"]))
        return 0
    if not args.case or not args.fixture_dir:
        raise SystemExit("--case and --fixture-dir are required")

    try:
        fixture = fixture_by_id(manifest, args.case)
        stages = load_stages(args.fixture_dir, args.case)
        colmap = None
        if args.colmap:
            try:
                colmap = parse_line(args.colmap, "golden-colmap")
            except (OSError, ValueError):
                colmap = None
        evaluation = evaluate(manifest, fixture, stages)
        status, errors = evaluation["status"], evaluation["errors"]
        markdown = render_markdown(fixture, evaluation, colmap)
        if args.json:
            args.json.write_text(
                json.dumps(
                    {
                        "case": args.case,
                        "status": status,
                        "first_failing_stage": evaluation["first_failing_stage"],
                        "errors": errors,
                        "stages": evaluation["stages"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
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
