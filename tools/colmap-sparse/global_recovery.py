"""Recover missing registered views with global SfM calibrated by the incremental map."""

from contextlib import closing

import pycolmap

from tracks import choose_exportable_seed, unambiguous_tracks


def remove_unsupported_frames(model):
    # A rotation estimate alone may register a GLOMAP camera without enough
    # unique landmarks to constrain translation. Deregistration also removes
    # its observations, so check again if another track lost its second view.
    changed = False
    while True:
        registered = [model.images[key] for key in model.reg_image_ids()]
        tracks, _ = unambiguous_tracks(model, registered)
        weak = [image.frame_id for image in registered if len(tracks[image.image_id]) < 8]
        if not weak:
            return changed
        for frame_id in weak:
            model.deregister_frame(frame_id)
        changed = True


def recover_globally(database, images, destination, initial):
    # Both solvers use the same single-camera video. Global positioning is
    # sensitive to focal length when the video contains long camera rotations.
    # Start it with the in-video bundle estimate from the trusted partial map.
    if len(initial.cameras) != 1:
        return {}
    camera = next(iter(initial.cameras.values()))
    options = pycolmap.GlobalPipelineOptions()
    options.num_threads = 2
    options.random_seed = 0
    destination.mkdir()
    with closing(pycolmap.Database.open(database)) as db:
        original = db.read_camera(camera.camera_id)
        db.update_camera(camera)
    try:
        recovered = pycolmap.global_mapping(database, images, destination, options=options)
    except RuntimeError as error:
        print(f"COLMAP global recovery failed: {error}", flush=True)
        return {}
    finally:
        with closing(pycolmap.Database.open(database)) as db:
            db.update_camera(original)

    initial_ids = set(initial.reg_image_ids())
    # Keep the connected incremental solution if a global component drops
    # its cameras or loses triangulated evidence while adding views.
    supported = {}
    for key, model in recovered.items():
        if not initial_ids < set(model.reg_image_ids()):
            continue
        changed = remove_unsupported_frames(model)
        if (initial_ids < set(model.reg_image_ids())
                and model.num_points3D() >= initial.num_points3D()
                and model.compute_num_observations() >= initial.compute_num_observations()):
            try:
                registered = sorted((model.images[i] for i in model.reg_image_ids()), key=lambda image: image.name)
                choose_exportable_seed(model, registered, check_global_parallax=True)
            except ValueError as error:
                print(f"Skipping global component without an exportable seed: {error}", flush=True)
                continue
            if changed:
                model.write(destination / str(key))
            supported[key] = model
    return supported
