"""Run with blender -b --factory-startup --python-exit-code 1 -P this_file."""
import sys
import unittest
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_rendered_scene import align, score


class DepthScoreTest(unittest.TestCase):
    def test_camera_only_similarity_recovers_imported_mesh_coordinates(self):
        rotation = np.array(Quaternion((.2, .7, .5), .8).to_matrix())
        offset = np.array([1., 3., -2.])
        scale = 2.5
        cv_to_local = np.diag([1., -1., -1.])
        # Corresponding Blender-imported coordinates for reconstruction (1, 2, 3).
        imported = np.array([1., 3., -2., 1.])
        cameras, poses = [], []
        for i in range(3):
            center = np.array([i, i / 2, .4 * i * i])
            r = np.array(Quaternion((0, 1, 0), .1 * i).to_matrix())
            pose = np.eye(4)
            pose[:3, :3] = rotation @ r.T @ cv_to_local
            pose[:3, 3] = scale * rotation @ center + offset
            poses.append(pose)
            cameras.append(dict(frame_index=i, rotation=r.flatten(), translation=-r @ center))
        point = align(cameras, np.array(poses)) @ imported
        np.testing.assert_allclose(point[:3], scale * rotation @ [1, 2, 3] + offset, atol=1e-5)

    def test_missing_and_misplaced_surfaces_lower_visible_ray_score(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, -2))
        capture = dict(width=12, height=12, horizontal_fov_radians=np.pi / 2)
        for right, depth, coverage, accurate in [(4, -2, 1, 1), (0, -2, .5, .5), (4, -2.5, 1, 0)]:
            mesh = BVHTree.FromPolygons(
                [(-4, -4, depth), (right, -4, depth), (right, 4, depth), (-4, 4, depth)],
                [(0, 1, 2), (0, 2, 3)], all_triangles=True,
            )
            result = score(bpy.context.scene, mesh, [np.eye(4)], capture, 12)
            self.assertEqual(result["reference_hits"], 144)
            self.assertEqual(result["visible_ray_coverage"], coverage)
            self.assertEqual(result["visible_rays_within_2pct_depth"], accurate)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DepthScoreTest)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise RuntimeError("Blender depth scoring regression")
