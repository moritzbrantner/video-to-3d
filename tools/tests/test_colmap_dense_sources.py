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

    def test_short_baseline_walks_get_two_sources_with_parallax(self):
        # 0.1 m steps subtend about 0.57 degrees at 10 m; only views >= 0.6 m apart reach 3 degrees.
        sources = reconstruct.select_sources(model([0.1 * k for k in range(12)]), limit=64)
        self.assertTrue(sources)
        for reference, chosen in sources.items():
            index = int(reference[:8])
            available = sum(abs(index - other) >= 6 for other in range(12))
            wide = [name for name in chosen if abs(index - int(name[:8])) >= 6]
            self.assertGreaterEqual(len(wide), min(2, available), (reference, chosen))

    def test_covisible_choice_is_kept_when_it_already_has_parallax(self):
        ranked = [(90, "a", True), (80, "b", False), (70, "c", True), (60, "d", True), (50, "e", True)]
        self.assertEqual(reconstruct.with_parallax(ranked, 3), ["a", "b", "c"])
        weak = [(90, "a", False), (80, "b", False), (70, "c", True), (60, "d", False), (50, "e", True)]
        self.assertEqual(reconstruct.with_parallax(weak, 3), ["a", "c", "e"])
        self.assertEqual(reconstruct.with_parallax([(9, "a", False), (8, "b", False)], 3), ["a", "b"])

    def test_without_any_parallax_the_covisible_choice_is_unchanged(self):
        sources = reconstruct.select_sources(model([0.01 * k for k in range(12)]), limit=64)
        # Every view shares all points, so ties rank by name: the four lowest other names.
        self.assertEqual(sources["00000005.png"], [f"{k:08d}.png" for k in (0, 1, 2, 3)])


if __name__ == "__main__":
    unittest.main()
