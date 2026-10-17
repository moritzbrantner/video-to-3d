#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
import math
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "video-to-3d/real-video-corpus/v1"
VALID_TIERS = {"quantitative", "field-canary"}
VALID_ASSET_MODES = {"provider", "manual"}
HANDHELD_APPROVAL = "https://github.com/moritzbrantner/video-to-3d/issues/74#issuecomment-6113239395"
APPROVED_SOURCES = {
    "approved-handheld-h1": {
        "sha256": "9d08559f3c18725848d0b53849dbc528c6ec803d4bc617cdafb2c16bdaf3304e",
        "owner_approval_url": HANDHELD_APPROVAL,
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
    },
    "approved-handheld-h3": {
        "sha256": "7885f38454d5ecd9bfc7f1e0a74d5522d8e2d1deea1d65338fbcd07871267fe4",
        "owner_approval_url": HANDHELD_APPROVAL,
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
    },
    "approved-controlled-room-tum": {
        "sha256": "3200d7a76f5b3307e33863d4b38dbad414527ce14084f4c1f6303a0f69eb794c",
        "owner_approval_url": "https://github.com/moritzbrantner/video-to-3d/issues/74#issuecomment-6117629409",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    },
}
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

    approved_cases: dict[str, dict[str, Any]] = {}
    if any(isinstance(case, dict) and (isinstance(case.get("id"), str) and case["id"] in APPROVED_SOURCES or isinstance(case.get("preparation"), dict)) for case in cases):
        try:
            approved_cases = _approved_cases()
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"approved preparation definitions unavailable: {error}")

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
        approved = isinstance(case_id, str) and case_id in APPROVED_SOURCES or isinstance(case.get("preparation"), dict)
        if asset.get("redistribute") is not False and not (approved and asset.get("redistribute") is True):
            errors.append(f"{prefix}.asset.redistribute must be false unless explicitly approved and licensed")
        if approved:
            errors.extend(_validate_preparation(case, prefix, approved_cases))

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


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _approved_cases() -> dict[str, dict[str, Any]]:
    # Public --manifest input cannot redefine approval. Reuse the repository's
    # pinned production corpus, independently protected by the source contract.
    path = Path(__file__).resolve().parents[1] / "benchmarks/real-video-corpus.json"
    canonical = load_manifest(path)
    approved: dict[str, dict[str, Any]] = {}
    for case in canonical["cases"]:
        if case.get("id") in APPROVED_SOURCES:
            if case["asset"]["sha256"] != APPROVED_SOURCES[case["id"]]["sha256"] or not isinstance(case.get("preparation"), dict):
                raise ValueError("canonical approved source/preparation mismatch")
            approved[case["id"]] = case
    if set(approved) != set(APPROVED_SOURCES):
        raise ValueError("canonical approved preparations missing")
    return approved


def _validate_preparation(case: dict[str, Any], prefix: str, approved: dict[str, dict[str, Any]] | None = None) -> list[str]:
    errors: list[str] = []
    if prefix != case.get("id"):
        prefix = f"{prefix} ({case.get('id')})"
    asset, preparation = case.get("asset"), case.get("preparation")
    if not isinstance(asset, dict):
        return [f"{prefix}.asset must be an object for approved preparation"]
    if approved is None:
        try:
            approved = _approved_cases()
        except (OSError, ValueError, KeyError, TypeError) as error:
            return [f"{prefix}: approved preparation definitions unavailable: {error}"]
    canonical = approved.get(case.get("id"))
    if canonical is None:
        errors.append(f"{prefix}: no approved preparation for this case")
    else:
        canonical_asset = canonical["asset"]
        for field in ("creator", "title", "citation", "source_url", "download_url", "license_url",
                      "owner_approval_url", "sha256", "byte_length", "expected_file",
                      "mode", "redistribute"):
            if asset.get(field) != canonical_asset.get(field):
                errors.append(f"{prefix}.asset.{field} differs from the approved canonical source")
        if not isinstance(preparation, dict):
            return errors + [f"{prefix}: approved canonical preparation must be an object"]
        for field in ("trim", "original_duration_seconds", "plan", "inventory"):
            if preparation.get(field) != canonical["preparation"].get(field):
                errors.append(f"{prefix}: {field} differs from the approved fixed preparation")
    if not isinstance(preparation, dict):
        return errors + [f"{prefix}: approved preparation must be an object"]
    source_policy = APPROVED_SOURCES.get(case.get("id"))
    if source_policy is None or any(asset.get(field) != source_policy[field] for field in ("sha256", "owner_approval_url")):
        errors.append(f"{prefix}: source is not an owner-approved preparation")
    for field in ("creator", "title", "license_url", "owner_approval_url", "download_url"):
        if not _nonempty_string(asset.get(field)) or (field.endswith("url") and not _https_url(asset[field])):
            errors.append(f"{prefix}.asset.{field} is required")
    digest = asset.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        errors.append(f"{prefix}.asset.sha256 must be a SHA256 digest")
    if not isinstance(asset.get("byte_length"), int) or isinstance(asset["byte_length"], bool) or asset["byte_length"] <= 0:
        errors.append(f"{prefix}.asset.byte_length must be positive")
    if asset.get("mode") != "manual" or source_policy is None or asset.get("license_url") != source_policy["license_url"]:
        errors.append(f"{prefix}: approved preparation requires its declared manual licensed source")
    for field in ("decoder_boundary", "trim", "plan", "inventory"):
        if not isinstance(preparation.get(field), dict):
            errors.append(f"{prefix}: {field} must be an object")
    if any(not isinstance(preparation.get(field), dict) for field in ("decoder_boundary", "trim", "plan", "inventory")):
        return errors
    boundary = preparation["decoder_boundary"]
    if boundary.get("kind") != "offline-ffmpeg" or boundary.get("browser_pixel_equivalence") is not False or boundary.get("sampling_authority") != "browser-plan-only":
        errors.append(f"{prefix}: offline preparation cannot claim browser pixel equivalence")
    trim = preparation.get("trim", {})
    start, end, duration = trim.get("start_seconds"), trim.get("end_seconds"), preparation.get("original_duration_seconds")
    if not all(_finite(v) for v in (start, end, duration)) or not 0 <= start < end <= duration:
        errors.append(f"{prefix}: invalid authorized trim")
    plan = preparation.get("plan", {})
    count = plan.get("sampleCount")
    times = plan.get("times", [])
    if count != 18 or not isinstance(times, list) or len(times) != 18 or any(not _finite(t) for t in times):
        errors.append(f"{prefix}: fixed18-frame plan required")
    elif all(_finite(v) for v in (start, end)) and (any(not 0 <= t < end-start for t in times) or times != sorted(set(times))):
        errors.append(f"{prefix}: sample requests must be ordered inside trim")
    if plan.get("analysisWidth") != 360 or plan.get("analysisHeight") != 270:
        errors.append(f"{prefix}: fixed360x270 analysis required")
    inventory = preparation.get("inventory", {})
    if inventory.get("metric_truth") is not False or not inventory.get("objects") or not inventory.get("architecture"):
        errors.append(f"{prefix}: independently observed nonmetric inventory required")
    for kind in ("objects", "architecture"):
        entries = inventory.get(kind)
        if not isinstance(entries, list):
            errors.append(f"{prefix}: inventory {kind} must be an array")
            continue
        for item in entries:
            if not isinstance(item, dict):
                errors.append(f"{prefix}: inventory item must be an object")
                continue
            visible = item.get("visible_samples")
            if not _nonempty_string(item.get("id")) or not isinstance(visible, list) or not visible or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < 18 for i in visible):
                errors.append(f"{prefix}: invalid inventory visibility")
            if kind == "objects" and (item.get("coverage") not in ("limited", "repeated") or not _nonempty_string(item.get("uncertainty"))):
                errors.append(f"{prefix}: object coverage uncertainty required")
    return errors


def _contained(root: Path, relative: Any) -> Path:
    if not _safe_relative_path(relative) or "\\" in relative or relative in (".", ""):
        raise ValueError("unsafe relative path")
    base = root.resolve()
    target = (base / relative).resolve()
    if target == base or not target.is_relative_to(base):
        raise ValueError("path escapes declared root")
    return target


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout


def _tool_version() -> str:
    return _run(["ffmpeg", "-version"]).splitlines()[0]


def _original(case: dict[str, Any], asset_root: Path) -> Path:
    original = _contained(asset_root, case["asset"]["expected_file"])
    if original.stat().st_size != case["asset"]["byte_length"] or _sha(original) != case["asset"]["sha256"]:
        raise ValueError("original integrity mismatch")
    return original


# Approved inputs share this browser-consumable deterministic carrier policy.
# Keep recipe identity in reference-cache keys when its implementation changes.
_DERIVATION_RECIPE = "lossless-vp9-webm-v1"


def _derive(original: Path, case: dict[str, Any], destination: Path) -> None:
    trim = case["preparation"]["trim"]
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(original), "-an", "-vf",
          f"trim=start={trim['start_seconds']}:end={trim['end_seconds']},setpts=PTS-STARTPTS",
          "-c:v", "libvpx-vp9", "-lossless", "1", "-cpu-used", "8", "-threads", "1", "-row-mt", "0",
          "-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1", "-y", str(destination)])


# Reuse verified immutable original-derived representations across receipt checks.
# Every call still hashes the actual original; cache keys include all semantic inputs.
_references: dict[str, tuple[list[tuple[int, Decimal, bytes]], str, int]] = {}


def _reference(original: Path, case: dict[str, Any], version: str) -> tuple[list[tuple[int, Decimal, bytes]], str, int]:
    key = json.dumps([case["asset"]["sha256"], case["preparation"], version, _DERIVATION_RECIPE], sort_keys=True)
    if key in _references:
        return _references[key]
    probe = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(original)]))
    pts = [Decimal(frame["best_effort_timestamp_time"]) for frame in probe["frames"]]
    trim, plan = case["preparation"]["trim"], case["preparation"]["plan"]
    start, end = Decimal(str(trim["start_seconds"])), Decimal(str(trim["end_seconds"]))
    candidates = [i for i, time in enumerate(pts) if start <= time < end]
    if not candidates:
        raise ValueError("authorized trim has no source frames")
    selected = [min(candidates, key=lambda i: (abs(pts[i]-(start+Decimal(str(time)))), pts[i], i)) for time in plan["times"]]
    with tempfile.TemporaryDirectory(prefix="video-handheld-reference-") as directory:
        base = Path(directory)
        expression = "+".join(f"eq(n,{i})" for i in selected)
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(original), "-vf", f"select='{expression}',scale={plan['analysisWidth']}:{plan['analysisHeight']}:flags=bilinear", "-fps_mode", "passthrough", "-start_number", "0", str(base / "%03d.ppm")])
        frames = [(ordinal, pts[ordinal], (base / f"{i:03d}.ppm").read_bytes()) for i, ordinal in enumerate(selected)]
        derived = base / "derived.webm"
        _derive(original, case, derived)
        result = frames, _sha(derived), derived.stat().st_size
    if len(_references) >= 2:
        _references.pop(next(iter(_references)))
    _references[key] = result
    return result


def _ppm_pixels(data: bytes) -> tuple[int, int, bytes]:
    position, tokens = 0, []
    while len(tokens) < 4:
        while position < len(data) and data[position:position+1].isspace():
            position += 1
        if data[position:position+1] == b"#":
            position = data.index(b"\n", position)+1
            continue
        end = position
        while end < len(data) and not data[end:end+1].isspace():
            end += 1
        if end == position:
            raise ValueError("incomplete P6 header")
        tokens.append(data[position:end])
        position = end
    if tokens[0] != b"P6" or tokens[3] != b"255" or not data[position:position+1].isspace():
        raise ValueError("expected 8-bit P6 frame")
    position += 2 if data[position:position+2] == b"\r\n" else 1
    width, height = int(tokens[1]), int(tokens[2])
    pixels = data[position:]
    if width <= 0 or height <= 0 or len(pixels) != width*height*3:
        raise ValueError("invalid P6 dimensions/pixel length")
    return width, height, pixels


def verify_prepared_case(case: dict[str, Any], asset_root: Path, prepared_root: Path) -> list[str]:
    try:
        metadata_errors = _validate_preparation(case, case["id"])
        if metadata_errors:
            return metadata_errors
        original = _original(case, Path(asset_root))
        base = _contained(Path(prepared_root), case["id"])
        receipt = load_manifest(_contained(base, "sampling.json"))
        version = _tool_version()
        if receipt.get("source") != case["asset"]["expected_file"] or receipt.get("original") != {"path": case["asset"]["expected_file"], "sha256": case["asset"]["sha256"], "byte_length": case["asset"]["byte_length"]}:
            raise ValueError("receipt original identity mismatch")
        if receipt.get("schema_version") != "video-to-3d/real-video-sampling/v1":
            raise ValueError("receipt schema mismatch")
        if receipt.get("case_id") != case["id"] or receipt.get("plan") != case["preparation"]["plan"]:
            raise ValueError("receipt case/plan mismatch")
        boundary = receipt.get("decoder_boundary", {})
        if any(boundary.get(k) != v for k, v in case["preparation"]["decoder_boundary"].items()) or boundary.get("tool_version") != version:
            raise ValueError("receipt decoder boundary/version mismatch")
        derived = receipt["derived"]
        if derived.get("source_sha256") != case["asset"]["sha256"] or derived.get("trim") != case["preparation"]["trim"]:
            raise ValueError("derived source/trim mismatch")
        references, derived_hash, derived_length = _reference(original, case, version)
        carrier = _contained(base, derived["path"])
        if derived.get("sha256") != derived_hash or _sha(carrier) != derived_hash or carrier.stat().st_size != derived_length or derived.get("byte_length") != derived_length:
            raise ValueError("derived bytes do not match authorized source trim")
        frames = receipt["frames"]
        if len(frames) != len(references):
            raise ValueError("frame count mismatch")
        for index, (frame, (ordinal, pts, pixels)) in enumerate(zip(frames, references)):
            relative = case["preparation"]["plan"]["times"][index]
            if frame.get("requested_relative_seconds") != relative or abs(frame.get("requested_source_seconds")-(relative+case["preparation"]["trim"]["start_seconds"])) > 1e-9 or frame.get("selected_source_ordinal") != ordinal or Decimal(str(frame.get("selected_source_pts"))) != pts:
                raise ValueError("frame original ordinal/PTS/request correspondence mismatch")
            actual = _contained(base, frame["path"]).read_bytes()
            if frame.get("sha256") != hashlib.sha256(actual).hexdigest() or _ppm_pixels(actual) != _ppm_pixels(pixels):
                raise ValueError("frame pixels do not match independently decoded original")
        return []
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ArithmeticError, subprocess.SubprocessError) as error:
        return [str(error)]


def prepare_case(case: dict[str, Any], asset_root: Path, prepared_root: Path) -> None:
    metadata_errors = _validate_preparation(case, case["id"])
    if metadata_errors:
        raise ValueError("preparation metadata invalid: " + "; ".join(metadata_errors))
    original = _original(case, asset_root)
    base = _contained(prepared_root, case["id"])
    # Reject symlink escapes before any destructive or generated write.
    if base.exists():
        errors = verify_prepared_case(case, asset_root, prepared_root)
        if not errors:
            return
        raise ValueError("existing preparation is invalid: " + "; ".join(errors))
    version = _tool_version()
    references, _, _ = _reference(original, case, version)
    base.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".handheld-", dir=base.parent) as temporary:
        staging = Path(temporary)
        (staging / "frames").mkdir()
        carrier = staging / "derived.webm"
        _derive(original, case, carrier)
        frames = []
        start = case["preparation"]["trim"]["start_seconds"]
        for index, (ordinal, pts, pixels) in enumerate(references):
            relative = case["preparation"]["plan"]["times"][index]
            path = f"frames/{index:03d}.ppm"
            (staging / path).write_bytes(pixels)
            frames.append({"path": path, "sha256": hashlib.sha256(pixels).hexdigest(), "requested_relative_seconds": relative, "requested_source_seconds": start+relative, "selected_source_ordinal": ordinal, "selected_source_pts": float(pts)})
        receipt = {"schema_version": "video-to-3d/real-video-sampling/v1", "case_id": case["id"], "source": case["asset"]["expected_file"], "original": {"path": case["asset"]["expected_file"], "sha256": case["asset"]["sha256"], "byte_length": case["asset"]["byte_length"]}, "plan": case["preparation"]["plan"], "derived": {"path": carrier.name, "sha256": _sha(carrier), "byte_length": carrier.stat().st_size, "source_sha256": case["asset"]["sha256"], "trim": case["preparation"]["trim"]}, "decoder_boundary": {**case["preparation"]["decoder_boundary"], "tool_version": version}, "frames": frames}
        (staging / "sampling.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        staging.rename(base)
    errors = verify_prepared_case(case, asset_root, prepared_root)
    if errors:
        raise ValueError("prepared verification failed: " + "; ".join(errors))


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


def validate_selected_case_approval(manifest: dict[str, Any], case_id: str) -> list[str]:
    """Guard pinned identities without changing the generic runner contract."""
    if case_id not in APPROVED_SOURCES:
        return []
    approved = _approved_cases()
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return [f"{case_id}: approved case selection requires a cases array"]
    selected = [case for case in cases if isinstance(case, dict) and case.get("id") == case_id]
    if len(selected) != 1:
        return [f"{case_id}: approved case selection must be unique (found {len(selected)})"]
    return _validate_preparation(selected[0], case_id, approved)


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
    parser.add_argument("--validate-case-approval", help="validate a selected pinned identity before runner dispatch")
    parser.add_argument("--prepare-case")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"real-video-manifest: {error}")
        return 1

    if args.validate_case_approval is not None:
        try:
            errors = validate_selected_case_approval(manifest, args.validate_case_approval)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors = [f"{args.validate_case_approval}: approved case definitions unavailable: {error}"]
        for error in errors:
            print(f"real-video-case-approval: {error}")
        return 1 if errors else 0

    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"real-video-manifest: {error}")
        return 1

    if args.prepare_case:
        try:
            case = next(case for case in manifest["cases"] if case["id"] == args.prepare_case)
            if not case.get("preparation") or args.asset_root is None or args.output_root is None:
                raise ValueError("approved preparation and explicit roots required")
            prepare_case(case, args.asset_root, args.output_root)
        except (StopIteration, OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"real-video-preparation: {error}")
            return 1
        print(f"real-video-preparation=ok case={args.prepare_case}")
        return 0

    if args.scheduled_cases:
        for case_id in scheduled_case_ids(manifest):
            print(case_id)
        return 0

    print(f"real-video-manifest=ok cases={len(manifest['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
