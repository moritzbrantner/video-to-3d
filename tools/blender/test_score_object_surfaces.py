"""Run with blender -b --factory-startup --python-exit-code 1 -P this_file."""
import sys
import unittest
from pathlib import Path

import numpy as np
from mathutils import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from object_surface_metrics import Surface, move, rigid_fit, surface_scores


class ObjectSurfaceScoreTest(unittest.TestCase):
    def test_aligning_pose_recovers_shape_but_preserves_scale(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [.2, .1, .5]])
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        cad = Surface(vertices, faces)
        offset = np.eye(4)
        offset[:3, :3] = np.array(Quaternion((.3, .1, .8), 1.1).to_matrix())
        offset[:3, 3] = [3., -1., .2]
        for scale in (1., 2.):
            mesh = Surface(move(vertices * scale, offset), faces)
            raw = surface_scores(mesh, cad, np.eye(4), 400, .01)
            self.assertGreater(raw["symmetric_mean_distance"], .8)
            fitted = rigid_fit(mesh, cad)
            rotation = fitted[:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            result = surface_scores(mesh, cad, fitted, 400, .01)
            if scale == 1.:
                self.assertLess(result["symmetric_mean_distance"], .004)
                self.assertGreater(result["f_score_at_threshold"], .99)
            else:
                self.assertGreater(result["symmetric_mean_distance"], .15)


if __name__ == "__main__":
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(ObjectSurfaceScoreTest)
    if not unittest.TextTestRunner().run(tests).wasSuccessful():
        raise RuntimeError("Blender object scoring regression")
