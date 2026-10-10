#!/usr/bin/env python3
"""COLMAP reference reconstruction for the golden fixtures, entirely in pycolmap.

Feature extraction, exact CPU matching, incremental mapping and text export all
run through the single pinned pycolmap (tools/requirements-colmap.txt), so the
database schema never crosses COLMAP versions. Mixing a distribution `colmap`
binary with pycolmap's database writes aborted the mapper on Ubuntu 26.04
(COLMAP 3.12 against a pycolmap 4.x database; #153).
"""

import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("camera_params", help="PINHOLE fx,fy,cx,cy")
    args = parser.parse_args()

    database = args.work / "database.db"
    sparse = args.work / "sparse"
    models = args.work / "models"
    sparse.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    pycolmap.set_random_seed(0)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = args.camera_params
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = 1
    extraction.use_gpu = False
    pycolmap.extract_features(
        str(database),
        str(args.images),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )

    # Exact CPU SIFT matching: approximate matching still varies with the seed.
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = 1
    matching.sift.cpu_brute_force_matcher = True
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = 0
    pycolmap.match_exhaustive(
        str(database),
        matching_options=matching,
        verification_options=verification,
        device=pycolmap.Device.cpu,
    )

    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = 1
    mapping.random_seed = 0
    mapping.mapper.init_min_tri_angle = 8
    mapping.ba_refine_focal_length = False
    mapping.ba_refine_principal_point = False
    mapping.ba_refine_extra_params = False
    try:
        reconstructions = pycolmap.incremental_mapping(
            str(database), str(args.images), str(sparse), options=mapping
        )
    except Exception as error:  # noqa: BLE001 - no model is reported as zero registrations
        print(f"COLMAP mapping failed: {error}", flush=True)
        reconstructions = {}

    for index, reconstruction in reconstructions.items():
        target = models / str(index)
        target.mkdir(parents=True, exist_ok=True)
        reconstruction.write_text(str(target))


if __name__ == "__main__":
    main()
