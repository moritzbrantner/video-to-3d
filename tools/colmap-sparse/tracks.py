"""Choose a camera gauge from the same unambiguous feature tracks exported to Rust."""

from itertools import combinations
from math import cos, dist, radians
from statistics import median


def unambiguous_tracks(model, registered):
    track_ids = {image.image_id: set() for image in registered}
    unambiguous_ids = []
    for point_id, point in model.points3D.items():
        image_ids = [element.image_id for element in point.track.elements]
        if len(set(image_ids)) != len(image_ids):
            # Multiple keypoints in the same image make ownership ambiguous.
            continue
        unambiguous_ids.append(point_id)
        for image_id in image_ids:
            track_ids[image_id].add(point_id)

    return track_ids, unambiguous_ids


def choose_exportable_seed(model, registered, *, check_global_parallax=False):
    track_ids, unambiguous_ids = unambiguous_tracks(model, registered)
    centers = {image.image_id: image.projection_center() for image in registered}
    descending_baselines = sorted(
        combinations(registered, 2),
        key=lambda pair: dist(centers[pair[0].image_id], centers[pair[1].image_id]),
        reverse=True,
    )
    for left, right in descending_baselines:
        common = track_ids[left.image_id] & track_ids[right.image_id]
        if len(common) < 12:
            continue
        if check_global_parallax:
            # Global rotation averaging can place many cameras while their
            # shared landmarks have little depth parallax. Seek another gauge
            # in the same map before replacing the supported incremental map.
            a, b = centers[left.image_id], centers[right.image_id]
            baseline_squared = dist(a, b) ** 2
            cosines = []
            for point_id in common:
                point = model.points3D[point_id].xyz
                da, db = dist(a, point), dist(b, point)
                if da > 0 and db > 0:
                    cosines.append((da * da + db * db - baseline_squared) / (2 * da * db))
            if len(cosines) < 12 or median(cosines) > cos(radians(1.5)):
                continue
        return left, right, unambiguous_ids
    raise ValueError("no camera pair has sufficient unambiguous triangulated track support")
