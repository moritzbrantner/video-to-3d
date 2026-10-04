#!/usr/bin/env python3
"""Use exact CPU SIFT matching for the small COLMAP reference fixtures."""

import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()

    pycolmap.set_random_seed(0)
    options = pycolmap.FeatureMatchingOptions()
    options.num_threads = 1
    options.sift.cpu_brute_force_matcher = True
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = 0
    pycolmap.match_exhaustive(
        str(args.database),
        matching_options=options,
        verification_options=verification,
        device=pycolmap.Device.cpu,
    )


if __name__ == "__main__":
    main()
