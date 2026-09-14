#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "video-to-3d/real-video-corpus/v1"
VALID_TIERS = {"quantitative", "field-canary"}
VALID_ASSET_MODES = {"provider", "manual"}
QUANTITATIVE_SCORING = "camera-aligned-uncropped-nearest-neighbor"


def _https_url(value: Any) -> bool:
    return isinstance(value, str) and urlparse(value).scheme == "https"


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _expected_quantitative_files(scene: str) -> set[str]:
    return {
        f"{scene}.ply",
        f"{scene}_COLMAP_SfM.log",
        f"{scene}_trans.txt",
        f"{scene}_mapping_reference.txt",
        f"{scene}.json",
    }


def scheduled_case_ids(manifest: dict[str, Any]) -> list[str]:
    return [
        case["id"]
        for case in manifest.get("cases", [])
        if isinstance(case, dict)
        and isinstance(case.get("automation"), dict)
        and case["automation"].get("scheduled") is True
        and isinstance(case.get("id"), str)
    ]


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        errors.append("sampling must be an object")
    else:
        fps = sampling.get("frames_per_second")
        max_frames = sampling.get("max_frames")
        if not isinstance(fps, (int, float)) or not (0 < fps <= 8):
            errors.append("sampling.frames_per_second must be in (0, 8]")
        if not isinstance(max_frames, int) or not (4 <= max_frames <= 120):
            errors.append("sampling.max_frames must be an integer in [4, 120]")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty array")
        return errors

    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue

        case_id = case.get("id")
        if not _nonempty_string(case_id):
            errors.append(f"{prefix}.id must be a non-empty string")
        elif case_id in seen_ids:
            errors.append(f"duplicate case id {case_id!r}")
        else:
            seen_ids.add(case_id)

        tier = case.get("tier")
        if tier not in VALID_TIERS:
            errors.append(f"{prefix}.tier must be one of {sorted(VALID_TIERS)}")

        asset = case.get("asset")
        if not isinstance(asset, dict):
            errors.append(f"{prefix}.asset must be an object")
            continue
        mode = asset.get("mode")
        if mode not in VALID_ASSET_MODES:
            errors.append(f"{prefix}.asset.mode must be one of {sorted(VALID_ASSET_MODES)}")
        if not _safe_relative_path(asset.get("expected_file")):
            errors.append(f"{prefix}.asset.expected_file must be a safe relative path")
        if not _https_url(asset.get("source_url")):
            errors.append(f"{prefix}.asset.source_url must be https")
        if asset.get("redistribute") is not False:
            errors.append(f"{prefix}.asset.redistribute must be false")

        automation = case.get("automation")
        scheduled = isinstance(automation, dict) and automation.get("scheduled") is True

        if mode == "provider":
            if asset.get("provider") != "tanks-and-temples":
                errors.append(f"{prefix}.asset.provider must be 'tanks-and-temples'")
            if not _nonempty_string(asset.get("scene")):
                errors.append(f"{prefix}.asset.scene must name the provider scene")
            if not _nonempty_string(asset.get("provider_file_id")):
                errors.append(f"{prefix}.asset.provider_file_id must be set")
            if not _nonempty_string(asset.get("checksum_file_id")):
                errors.append(f"{prefix}.asset.checksum_file_id must be set")
            if asset.get("integrity") != "provider-md5":
                errors.append(f"{prefix}.asset.integrity must be 'provider-md5'")
            if not _https_url(asset.get("license_url")):
                errors.append(f"{prefix}.asset.license_url must be https")
        elif scheduled:
            errors.append(f"{prefix}: manual assets cannot be scheduled")

        evaluation = case.get("evaluation")
        if not isinstance(evaluation, dict):
            errors.append(f"{prefix}.evaluation must be an object")
            continue
        if tier == "quantitative":
            scene = asset.get("scene") if isinstance(asset.get("scene"), str) else ""
            if evaluation.get("ground_truth") != "laser-scan":
                errors.append(f"{prefix}.evaluation.ground_truth must be 'laser-scan'")
            if evaluation.get("reference_reconstruction") != "COLMAP":
                errors.append(f"{prefix}.evaluation.reference_reconstruction must be 'COLMAP'")
            if evaluation.get("camera_reference") is not True:
                errors.append(f"{prefix}.evaluation.camera_reference must be true")
            if not _https_url(evaluation.get("data_source_url")):
                errors.append(f"{prefix}.evaluation.data_source_url must be https")
            if not _nonempty_string(evaluation.get("provider_bundle_id")):
                errors.append(f"{prefix}.evaluation.provider_bundle_id must be set")
            expected_files = evaluation.get("expected_files")
            if not isinstance(expected_files, list) or set(expected_files) != _expected_quantitative_files(
                scene
            ):
                errors.append(
                    f"{prefix}.evaluation.expected_files must contain the scene ground truth, COLMAP camera reference, alignment, mapping, and crop files"
                )
            tau = evaluation.get("distance_tau")
            if not isinstance(tau, (int, float)) or tau <= 0:
                errors.append(f"{prefix}.evaluation.distance_tau must be positive")
            if evaluation.get("scoring") != QUANTITATIVE_SCORING:
                errors.append(f"{prefix}.evaluation.scoring must be {QUANTITATIVE_SCORING!r}")
            if evaluation.get("gate") != "evidence-only-until-baseline":
                errors.append(f"{prefix}.evaluation.gate must remain evidence-only until baselined")
        elif evaluation.get("gate") != "manual-field-canary":
            errors.append(f"{prefix}.evaluation.gate must be 'manual-field-canary'")

    return errors


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the real-video benchmark corpus manifest")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("benchmarks/real-video-corpus.json"),
    )
    parser.add_argument(
        "--scheduled-cases",
        action="store_true",
        help="print validated scheduled case ids, one per line",
    )
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"real-video-manifest: {error}")
        return 1

    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"real-video-manifest: {error}")
        return 1

    if args.scheduled_cases:
        for case_id in scheduled_case_ids(manifest):
            print(case_id)
        return 0

    print(f"real-video-manifest=ok cases={len(manifest['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
