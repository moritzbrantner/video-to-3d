#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

SCENE_FILE_IDS = {
    "Church": "0B-ePgl6HF260dnlGMkFkNlpibG8",
    "Ignatius": "0B-ePgl6HF260T19oUTIyUTRwTE0",
}
VIDEO_CHECKSUM_FILE_ID = "0B-ePgl6HF260M2h5Q3o1bGdpc1U"


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "-O",
            str(destination),
        ],
        check=True,
    )


def expected_checksum(checksum_file: Path, scene: str) -> str:
    expected_name = f"{scene}.mp4"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and Path(parts[-1]).name == expected_name:
            return parts[0].lower()
    raise RuntimeError(f"official checksum list has no entry for {expected_name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire one checksum-verified Tanks and Temples training video"
    )
    parser.add_argument("--scene", choices=sorted(SCENE_FILE_IDS), required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(".benchmark-data/tanks-and-temples"),
    )
    args = parser.parse_args()

    checksum_file = args.asset_root / "video_set_md5.chk"
    if not checksum_file.exists():
        download(VIDEO_CHECKSUM_FILE_ID, checksum_file)

    destination = args.asset_root / "videos" / f"{args.scene}.mp4"
    expected = expected_checksum(checksum_file, args.scene)
    if destination.exists() and md5(destination) == expected:
        print(f"tanks-and-temples={args.scene} checksum=ok cached=true")
        return 0

    destination.unlink(missing_ok=True)
    download(SCENE_FILE_IDS[args.scene], destination)
    actual = md5(destination)
    if actual != expected:
        destination.unlink(missing_ok=True)
        print(
            f"Tanks and Temples checksum mismatch for {args.scene}: expected {expected}, got {actual}",
            file=sys.stderr,
        )
        return 1

    print(f"tanks-and-temples={args.scene} checksum=ok cached=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
