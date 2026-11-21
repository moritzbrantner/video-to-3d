import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


source = Path(__file__).resolve().parents[1] / "colmap-dense" / "reconstruct.py"
spec = importlib.util.spec_from_file_location("colmap_dense_reconstruct", source)
reconstruct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruct)


def model(xs):
    """Cameras on a line at the given x positions, all seeing 12 points 10 m ahead."""
    points = {i: SimpleNamespace(xyz=(i * 0.1 - 0.6, 0.0, 10.0)) for i in range(12)}
    images = {
        k: SimpleNamespace(
            name=f"{k:08d}.png",
            projection_center=lambda x=x: (x, 0.0, 0.0),
            points2D=[SimpleNamespace(point3D_id=i, has_point3D=lambda: True) for i in points],
        )
        for k, x in enumerate(xs)
    }
    return SimpleNamespace(images=images, points3D=points)


class SourceSelectionTests(unittest.TestCase):
    def test_angle_is_measured_at_the_shared_points(self):
        angle = reconstruct.median_triangulation_angle(
            {"a": (0.0, 0.0, 0.0), "b": (1.0, 0.0, 0.0)}, lambda ids: [(0.0, 0.0, 10.0)] * len(ids), "a", "b", [1, 2])
        self.assertAlmostEqual(angle, 5.7106, places=3)

    def test_adjacent_short_baseline_views_are_not_sources(self):
        # 0.1 m steps subtend about 0.57 degrees at 10 m; only views >= 0.6 m apart reach 3 degrees.
        sources = reconstruct.select_sources(model([0.1 * k for k in range(12)]), limit=64)
        self.assertTrue(sources)
        for reference, chosen in sources.items():
            for name in chosen:
                gap = abs(int(reference[:8]) - int(name[:8]))
                self.assertGreaterEqual(gap, 6, (reference, chosen))

    def test_without_enough_parallax_nothing_is_offered_to_patchmatch(self):
        self.assertEqual(reconstruct.select_sources(model([0.01 * k for k in range(12)]), limit=64), {})


if __name__ == "__main__":
    unittest.main()
