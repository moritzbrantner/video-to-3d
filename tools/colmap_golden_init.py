#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path


def load_images(database: Path) -> list[tuple[int, str]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT image_id, name FROM images ORDER BY name, image_id"
        ).fetchall()
    finally:
        connection.close()
    return [(int(image_id), str(name)) for image_id, name in rows]


def candidate_pairs(
    images: list[tuple[int, str]],
) -> list[tuple[int, int, str, str]]:
    if len(images) < 2:
        raise ValueError("COLMAP golden reference needs at least two images")

    # The golden matrix intentionally includes a revisit trajectory where the
    # final frame returns toward the initial viewpoint. A first/last seed is
    # therefore not a reliable baseline. The midpoint is the furthest part of
    # that trajectory and is also a healthy baseline for the monotonic cases.
    midpoint = len(images) // 2
    raw_indices = [
        (0, midpoint),
        (0, min(len(images) - 1, midpoint + 1)),
        (1, min(len(images) - 1, midpoint + 1)),
        (len(images) // 4, min(len(images) - 1, (3 * len(images)) // 4)),
        (0, len(images) - 1),
    ]

    pairs: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for left_index, right_index in raw_indices:
        if left_index == right_index:
            continue
        left = images[left_index]
        right = images[right_index]
        identity = (left[0], right[0])
        if identity in seen:
            continue
        seen.add(identity)
        pairs.append((left[0], right[0], left[1], right[1]))
    return pairs


def render_plan(database: Path) -> str:
    images = load_images(database)
    lines = [f"count\t{len(images)}"]
    lines.extend(
        f"pair\t{left_id}\t{right_id}\t{left_name}\t{right_name}"
        for left_id, right_id, left_name, right_name in candidate_pairs(images)
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(render_plan(args.database))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
