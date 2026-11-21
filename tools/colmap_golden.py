#!/usr/bin/env python3
"""COLMAP reference reconstruction for the golden fixtures, entirely in pycolmap.

Feature extraction, exact CPU matching, incremental mapping and text export all
run through the single pinned pycolmap (tools/requirements-colmap.txt), so the
database schema never crosses COLMAP versions. Mixing a distribution `colmap`
binary with pycolmap's database writes aborted the mapper on Ubuntu 26.04
(COLMAP 3.12 against a pycolmap 4.x database; #153).
"""

import argparse
from contextlib import closing
from pathlib import Path

import pycolmap


def matching_workload(database):
    with closing(pycolmap.Database.open(database)) as db:
        counts = [db.num_keypoints_for_image(image.image_id) for image in db.read_all_images()]
    # Every exhaustive image pair compares each descriptor in one image with every
    # descriptor in the other. Subtract self-pairs and count each pair once.
    comparisons = (sum(counts) ** 2 - sum(count ** 2 for count in counts)) // 2
    return len(counts), comparisons


def refine_revisits(database, work, temporal_models, propose_revisits, match_options, map_into):
    if propose_revisits is None:
        return {}
    model = max(temporal_models.values(), key=lambda item: (item.num_reg_images(), item.num_points3D()))
    try:
        pairs = propose_revisits(database, model)
        if not pairs:
            return {}
        print(f"Verifying {len(pairs)} nonlocal camera pairs for loop closure.", flush=True)
        pair_list = work / "revisit-pairs.txt"
        pair_list.write_text("".join(f"{left} {right}\n" for left, right in pairs))
        imported = pycolmap.ImportedPairingOptions()
        imported.match_list_path = pair_list
        pycolmap.match_image_pairs(str(database), pairing_options=imported, **match_options)
        return map_into(work / "revisited")
    except (RuntimeError, OSError, ValueError) as error:
        print(f"COLMAP revisit refinement failed, retaining the complete temporal model: {error}", flush=True)
        return {}


def solve(
    images, work, camera_params, *, refine_focal=False, feature_threads=1, match_threads=1,
    exhaustive_comparison_budget=None, propose_revisits=None, recover_partial=None, diagnostics=None,
):
    """Run a single-camera CPU solve on local decoded frames."""
    database = work / "database.db"
    sparse = work / "sparse"

    pycolmap.set_random_seed(0)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_PINHOLE" if refine_focal else "PINHOLE"
    reader.camera_params = camera_params
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = feature_threads
    extraction.use_gpu = False
    pycolmap.extract_features(
        str(database),
        str(images),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )

    # Exact CPU SIFT matching: approximate matching still varies with the seed.
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = match_threads
    matching.sift.cpu_brute_force_matcher = True
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = 0
    match_options = dict(
        matching_options=matching,
        verification_options=verification,
        device=pycolmap.Device.cpu,
    )

    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = 1
    mapping.random_seed = 0
    mapping.mapper.init_min_tri_angle = 8
    mapping.ba_refine_focal_length = refine_focal
    mapping.ba_refine_principal_point = False
    mapping.ba_refine_extra_params = False

    def map_into(directory):
        directory.mkdir(parents=True, exist_ok=True)
        try:
            return pycolmap.incremental_mapping(
                str(database), str(images), str(directory), options=mapping
            )
        except Exception as error:  # noqa: BLE001 - no model is reported as zero registrations
            print(f"COLMAP mapping failed: {error}", flush=True)
            return {}

    def largest_model(models):
        return max(((m.num_reg_images(), m.num_points3D()) for m in models.values()), default=(0, 0))

    if diagnostics is not None:
        diagnostics.update(mapper="incremental", pairing="exhaustive")
    temporal_models = {}
    temporal = work / "temporal"
    if exhaustive_comparison_budget is not None:
        count, comparisons = matching_workload(database)
        if comparisons > exhaustive_comparison_budget:
            print(f"Trying temporal pairs for {count} sampled frames ({comparisons:,} exhaustive descriptor comparisons).", flush=True)
            try:
                pycolmap.match_sequential(str(database), **match_options)
                temporal_models = map_into(temporal)
            except RuntimeError as error:
                print(f"COLMAP temporal matching failed: {error}", flush=True)
            registered, _ = largest_model(temporal_models)
            if registered == count:
                if diagnostics is not None:
                    diagnostics.update(pairing="temporal", incremental_registered_frames=registered)
                revisited = refine_revisits(database, work, temporal_models, propose_revisits, match_options, map_into)
                if largest_model(revisited)[0] == count:
                    if diagnostics is not None:
                        diagnostics["pairing"] = "temporal_revisits"
                    (work / "revisited").rename(sparse)
                    return revisited
                temporal.rename(sparse)
                return temporal_models
            print(f"Temporal pairs registered {registered}/{count} frames; adding exhaustive pairs.", flush=True)

    pycolmap.match_exhaustive(str(database), **match_options)
    all_models = map_into(sparse)
    if temporal_models and largest_model(temporal_models) > largest_model(all_models):
        sparse.rename(work / "exhaustive")
        temporal.rename(sparse)
        all_models = temporal_models
        if diagnostics is not None:
            diagnostics["pairing"] = "temporal"
    if diagnostics is not None:
        diagnostics["incremental_registered_frames"] = largest_model(all_models)[0]
    if recover_partial is not None and all_models:
        initial = max(all_models.values(), key=lambda item: (item.num_reg_images(), item.num_points3D()))
        with closing(pycolmap.Database.open(database)) as db:
            frame_count = db.num_images()
        if initial.num_reg_images() < frame_count:
            print(f"Trying calibrated global SfM after registering {initial.num_reg_images()}/{frame_count} frames.", flush=True)
            recovered = recover_partial(database, images, work / "global", initial)
            if recovered:
                if diagnostics is not None:
                    diagnostics["mapper"] = "global"
                sparse.rename(work / "partial")
                (work / "global").rename(sparse)
                return recovered
    return all_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("camera_params", help="PINHOLE fx,fy,cx,cy")
    args = parser.parse_args()

    models = args.work / "models"
    models.mkdir(parents=True, exist_ok=True)
    reconstructions = solve(args.images, args.work, args.camera_params)
    for index, reconstruction in reconstructions.items():
        target = models / str(index)
        target.mkdir(parents=True, exist_ok=True)
        reconstruction.write_text(str(target))


if __name__ == "__main__":
    main()
