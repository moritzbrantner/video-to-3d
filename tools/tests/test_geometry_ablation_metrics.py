import importlib.util
from pathlib import Path
import unittest


source = Path(__file__).resolve().parents[1] / "geometry-diagnostics" / "ablation_metrics.py"
spec = importlib.util.spec_from_file_location("ablation_metrics", source)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class RayClassificationTests(unittest.TestCase):
    def test_distinguishes_missing_false_positive_and_hole(self):
        self.assertEqual(metrics.classify_ray(2.0, None), "missing")
        self.assertEqual(metrics.classify_ray(2.0, 2.09), "correct")
        self.assertEqual(metrics.classify_ray(2.0, 1.85), "false_positive_in_front")
        self.assertEqual(metrics.classify_ray(2.0, 2.4), "behind_truth")
        self.assertEqual(metrics.classify_ray(2.0, 1.97, tolerance=metrics.TIGHT_TOLERANCE_M), "correct")

    def test_invalid_measurements_are_not_scored(self):
        with self.assertRaises(ValueError):
            metrics.classify_ray(0.0, 1.0)

    def test_summary_fractions_cover_every_valid_ray(self):
        summary = metrics.ray_summary({"correct": 6, "false_positive_in_front": 1, "behind_truth": 1, "missing": 2})
        self.assertAlmostEqual(sum(summary.values()), 1.0)
        self.assertAlmostEqual(summary["missing"], 0.2)
        with self.assertRaises(ValueError):
            metrics.ray_summary({})

    def test_percentile_matches_linear_interpolation(self):
        self.assertEqual(metrics.percentile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(metrics.percentile([0, 10], 90), 9.0)
        self.assertIsNone(metrics.percentile([], 90))


class AttributionTests(unittest.TestCase):
    # Held-out correct-ray fractions measured on the pinned TUM run (2026-11-21).
    TUM = {
        "reference_pose_measured_depth_tsdf": 0.8399,
        "estimated_pose_measured_depth_tsdf": 0.8379,
        "estimated_pose_estimated_depth_tsdf": 0.5789,
        "estimated_pose_estimated_depth_rust": 0.5441,
    }

    def test_tum_loss_is_first_material_at_depth_estimation(self):
        result = metrics.attribute_loss(self.TUM)
        self.assertEqual(result["largest_loss_stage"], "depth_estimation")
        self.assertEqual(result["earliest_material_loss_stage"], "depth_estimation")
        self.assertAlmostEqual(result["losses"]["camera_poses"], 0.002, places=3)
        self.assertAlmostEqual(result["losses"]["surface_assembly"], 0.0348, places=3)

    def test_pose_loss_is_reported_before_a_larger_later_loss(self):
        result = metrics.attribute_loss({
            "reference_pose_measured_depth_tsdf": 0.9,
            "estimated_pose_measured_depth_tsdf": 0.8,
            "estimated_pose_estimated_depth_tsdf": 0.6,
            "estimated_pose_estimated_depth_rust": 0.55,
        })
        self.assertEqual(result["earliest_material_loss_stage"], "camera_poses")
        self.assertEqual(result["largest_loss_stage"], "depth_estimation")

    def test_requires_every_controlled_combination(self):
        with self.assertRaises(ValueError):
            metrics.attribute_loss({"reference_pose_measured_depth_tsdf": 0.8})

    def test_markdown_table_formats_units(self):
        table = metrics.markdown_table(
            [{"name": "rust", "share": 0.5441, "error": 0.0269, "count": 108445, "none": None}],
            [("name", "Surface", "s"), ("share", "Correct", "%"), ("error", "cm", "cm"),
             ("count", "Triangles", "int"), ("none", "Missing", "%")],
        )
        self.assertIn("| rust | 54.4% | 2.7 | 108,445 | – |", table)


if __name__ == "__main__":
    unittest.main()
