"""Propose a bounded number of nonlocal matches from a registered video trajectory."""

from contextlib import closing

import numpy as np
import pycolmap


def revisit_pairs(database, model):
    images = sorted(model.images.values(), key=lambda image: image.name)
    centers = np.array([image.projection_center() for image in images])
    directions = np.array([image.cam_from_world().rotation.matrix()[2] for image in images])
    depths = []
    for image, center in zip(images, centers):
        distances = [
            np.linalg.norm(model.points3D[point.point3D_id].xyz - center)
            for point in image.points2D if point.has_point3D()
        ]
        depths.append(np.median(distances) if distances else 0.0)

    proposals = set()
    with closing(pycolmap.Database.open(database)) as db:
        for i, image in enumerate(images):
            angles = np.arccos(np.clip(directions @ directions[i], -1, 1))
            depth = np.minimum(depths[i], depths)
            distance = np.linalg.norm(centers - centers[i], axis=1)
            separation = np.divide(distance, depth, out=np.full_like(distance, np.inf), where=depth > 0)
            # SIFT verifies proposed revisits. Similar view directions and a
            # baseline under half the observed scene depth favor image overlap.
            nearby = np.flatnonzero((angles < np.pi / 4) & (separation < 0.5))
            ranked = nearby[np.argsort((angles + separation)[nearby], kind="stable")]
            count = 0
            for j in ranked:
                other = images[j]
                if abs(i - j) <= 10 or db.exists_matches(image.image_id, other.image_id):
                    continue
                proposals.add(tuple(sorted((image.name, other.name))))
                count += 1
                if count == 2:
                    break
    return sorted(proposals)
