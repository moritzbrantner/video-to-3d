"""Whole-object surface distances and evaluation-only rigid pose alignment in Blender."""
from itertools import permutations, product

import numpy as np
from mathutils.bvhtree import BVHTree


class Surface:
    def __init__(self, vertices, faces):
        self.triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(self.triangles[:, 1] - self.triangles[:, 0],
                                        self.triangles[:, 2] - self.triangles[:, 0]), axis=1)
        if not np.isfinite(areas).all() or areas.sum() <= 0:
            raise ValueError("surface must have finite, positive triangle area")
        self.weights = areas / areas.sum()
        self.tree = BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)

    def samples(self, count, seed):
        rng = np.random.default_rng(seed)
        triangles = self.triangles[rng.choice(len(self.triangles), count, p=self.weights)]
        u, v = rng.random((2, count, 1))
        u = np.sqrt(u)
        return (1 - u) * triangles[:, 0] + u * (1 - v) * triangles[:, 1] + u * v * triangles[:, 2]

    def nearest(self, points):
        return np.array([tuple(self.tree.find_nearest(p)[0]) for p in points])


def move(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def pose_matrix(rotation, source_center, target_center):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def point_fit(points, targets):
    center, target_center = points.mean(axis=0), targets.mean(axis=0)
    u, _, vt = np.linalg.svd((points - center).T @ (targets - target_center))
    sign = np.linalg.det(vt.T @ u.T)
    return pose_matrix(vt.T @ np.diag([1., 1., sign]) @ u.T, center, target_center)


def correspondences(mesh, cad, mesh_points, cad_points, pose):
    """Find exact closest surface points in both directions, with equal sample counts."""
    forward = move(mesh_points, pose)
    cad_matches = cad.nearest(forward)
    mesh_matches = mesh.nearest(move(cad_points, np.linalg.inv(pose)))
    return np.vstack([mesh_points, mesh_matches]), np.vstack([cad_matches, cad_points])


def principal_axes(points):
    return np.linalg.eigh(np.cov(points.T))[1]


def rigid_fit(mesh, cad):
    """Multi-start symmetric ICP, preserving scale, handedness and every vertex."""
    a, b = mesh.samples(1200, 123), cad.samples(1200, 245)
    axes_a, axes_b = principal_axes(a), principal_axes(b)
    starts = [np.eye(4), pose_matrix(np.eye(3), a.mean(axis=0), b.mean(axis=0))]
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = (axes_b[:, perm] * signs) @ axes_a.T
            if np.linalg.det(rotation) > 0:
                starts.append(pose_matrix(rotation, a.mean(axis=0), b.mean(axis=0)))

    def refine(pose, sample_count, steps):
        best_error = np.inf
        best_pose = pose
        for iteration in range(steps + 1):
            src, dst = correspondences(mesh, cad, a[:sample_count], b[:sample_count], pose)
            error = np.mean(np.sum((move(src, pose) - dst) ** 2, axis=1))
            if error < best_error:
                best_pose, best_error = pose, error
            if iteration == steps:
                break
            next_pose = point_fit(src, dst)
            if np.allclose(next_pose, pose, rtol=0, atol=1e-6):
                break
            pose = next_pose
        return best_error, best_pose

    coarse = sorted((refine(start, 96, 2) for start in starts), key=lambda item: item[0])
    return min((refine(pose, len(a), 22) for _, pose in coarse[:3]), key=lambda item: item[0])[1]


def summarize(distances):
    return {"median": float(np.median(distances)), "p90": float(np.quantile(distances, .90)),
            "mean": float(np.mean(distances))}


def surface_scores(mesh, cad, pose, count, threshold):
    # The held-out samples use independent seeds from the fit and from each other.
    a = move(mesh.samples(count, 8191), pose)
    b = move(cad.samples(count, 5243), np.linalg.inv(pose))
    forward = np.linalg.norm(a - cad.nearest(a), axis=1)
    reverse = np.linalg.norm(b - mesh.nearest(b), axis=1)
    precision, recall = float(np.mean(forward <= threshold)), float(np.mean(reverse <= threshold))
    return {"mesh_to_cad": summarize(forward), "cad_to_mesh": summarize(reverse),
            "symmetric_mean_distance": float((forward.mean() + reverse.mean()) / 2),
            "precision_at_threshold": precision, "recall_at_threshold": recall,
            "f_score_at_threshold": 2 * precision * recall / (precision + recall) if precision + recall else 0}
