#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "-O",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(destination)


def expected_checksum(checksum_file: Path, scene: str) -> str:
    expected_name = f"{scene}.mp4"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and Path(parts[-1]).name == expected_name:
            return parts[0].lower()
    raise RuntimeError(f"official checksum list has no entry for {expected_name}")


def load_case(manifest_path: Path, case_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for case in manifest.get("cases", []):
        if case.get("id") == case_id:
            if case.get("tier") != "quantitative":
                raise RuntimeError(f"{case_id} is not a quantitative provider case")
            asset = case.get("asset", {})
            if asset.get("provider") != "tanks-and-temples":
                raise RuntimeError(f"{case_id} is not a Tanks and Temples provider case")
            return case
    raise RuntimeError(f"unknown real-video benchmark case {case_id}")


def ensure_video(case: dict[str, Any], asset_root: Path) -> Path:
    asset = case["asset"]
    scene = asset["scene"]
    checksum_file = asset_root / "video_set_md5.chk"
    if not checksum_file.exists():
        download(asset["checksum_file_id"], checksum_file)

    destination = asset_root / "videos" / f"{scene}.mp4"
    expected = expected_checksum(checksum_file, scene)
    if destination.exists() and md5(destination) == expected:
        print(f"tanks-and-temples={scene} video-checksum=ok cached=true")
        return destination

    destination.unlink(missing_ok=True)
    download(asset["provider_file_id"], destination)
    actual = md5(destination)
    if actual != expected:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Tanks and Temples checksum mismatch for {scene}: expected {expected}, got {actual}"
        )

    print(f"tanks-and-temples={scene} video-checksum=ok cached=false")
    return destination


def extract_evaluation_files(
    archive: Path,
    scene_directory: Path,
    expected_files: list[str],
) -> None:
    scene_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        by_name: dict[str, list[str]] = {}
        for member in bundle.namelist():
            by_name.setdefault(Path(member).name, []).append(member)

        for filename in expected_files:
            matches = by_name.get(filename, [])
            if len(matches) != 1:
                raise RuntimeError(
                    f"evaluation bundle expected exactly one {filename}, found {len(matches)}"
                )
            destination = scene_directory / filename
            temporary = destination.with_name(f"{destination.name}.part")
            temporary.unlink(missing_ok=True)
            with bundle.open(matches[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)


def ensure_evaluation(case: dict[str, Any], asset_root: Path) -> Path:
    scene = case["asset"]["scene"]
    evaluation = case["evaluation"]
    expected_files = list(evaluation["expected_files"])
    scene_directory = asset_root / "evaluation" / scene
    if all((scene_directory / filename).is_file() for filename in expected_files):
        print(f"tanks-and-temples={scene} evaluation-data=ok cached=true")
        return scene_directory

    archive = asset_root / "evaluation" / "training-evaluation.zip"
    for attempt in range(2):
        if not archive.exists():
            download(evaluation["provider_bundle_id"], archive)
        try:
            extract_evaluation_files(archive, scene_directory, expected_files)
            print(f"tanks-and-temples={scene} evaluation-data=ok cached=false")
            return scene_directory
        except (zipfile.BadZipFile, RuntimeError):
            if attempt == 1:
                raise
            archive.unlink(missing_ok=True)

    raise RuntimeError("unreachable evaluation acquisition state")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire a checksum-verified Tanks and Temples video and optional evaluation data"
    )
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/real-video-corpus.json"),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(".benchmark-data/tanks-and-temples"),
    )
    parser.add_argument("--with-evaluation", action="store_true")
    args = parser.parse_args()

    try:
        case = load_case(args.manifest, args.case)
        ensure_video(case, args.asset_root)
        if args.with_evaluation:
            ensure_evaluation(case, args.asset_root)
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"tanks-and-temples: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
