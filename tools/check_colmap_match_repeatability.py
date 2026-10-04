#!/usr/bin/env python3
"""Compare exact matches and verified geometry from independent COLMAP runs."""

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3


def compare(reference: Path, candidate: Path) -> list[str]:
    differences = []
    with closing(sqlite3.connect(reference.resolve().as_uri() + "?mode=ro", uri=True)) as left:
        with closing(sqlite3.connect(candidate.resolve().as_uri() + "?mode=ro", uri=True)) as right:
            for table in ("matches", "two_view_geometries"):
                query = f"SELECT * FROM {table} ORDER BY pair_id"
                if left.execute(query).fetchall() != right.execute(query).fetchall():
                    differences.append(f"COLMAP {table} differ between quality and runtime runs")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        differences = compare(args.reference, args.candidate)
    except sqlite3.Error as error:
        print(f"golden-repeatability: {error}")
        return 1
    for difference in differences:
        print(f"golden-repeatability: {difference}")
    if differences:
        return 1
    print("golden-repeatability: exact matches and verified geometry are identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
