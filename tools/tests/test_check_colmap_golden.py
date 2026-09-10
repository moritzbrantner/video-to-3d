import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "check_colmap_golden.py"
spec = importlib.util.spec_from_file_location("check_colmap_golden", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GoldenGateTests(unittest.TestCase):
    def metrics(self):
        rust = {
            "case": "revisit",
            "registered_images": "7",
            "points": "380",
            "median_reprojection_error_pixels": "0.25",
            "normalized_pose_rmse": "0.08",
        }
        colmap = {
            "case": "revisit",
            "registered_images": "8",
            "points": "500",
            "mean_reprojection_error_pixels": "0.4",
        }
        return rust, colmap

    def test_accepts_supported_fixture(self):
        rust, colmap = self.metrics()
        self.assertEqual(
            module.evaluate(
                rust,
                colmap,
                case="revisit",
                expected_images=8,
                max_normalized_pose_rmse=0.25,
            ),
            [],
        )

    def test_accepts_independent_rust_camera_coverage(self):
        rust, colmap = self.metrics()
        rust["registered_images"] = "5"
        self.assertEqual(
            module.evaluate(
                rust,
                colmap,
                case="revisit",
                expected_images=8,
                max_normalized_pose_rmse=0.25,
            ),
            [],
        )

    def test_rejects_wrong_case_identity(self):
        rust, colmap = self.metrics()
        rust["case"] = "lateral"
        errors = module.evaluate(
            rust,
            colmap,
            case="revisit",
            expected_images=8,
            max_normalized_pose_rmse=0.25,
        )
        self.assertTrue(any("expected" in error and "case" in error.lower() for error in errors))

    def test_rejects_pose_drift_even_when_reprojection_is_good(self):
        rust, colmap = self.metrics()
        rust["normalized_pose_rmse"] = "0.31"
        errors = module.evaluate(
            rust,
            colmap,
            case="revisit",
            expected_images=8,
            max_normalized_pose_rmse=0.25,
        )
        self.assertTrue(any("pose RMSE" in error for error in errors))

    def test_rejects_camera_coverage_regression(self):
        rust, colmap = self.metrics()
        rust["registered_images"] = "3"
        errors = module.evaluate(
            rust,
            colmap,
            case="revisit",
            expected_images=8,
            max_normalized_pose_rmse=0.25,
        )
        self.assertTrue(any("registered 3/8" in error for error in errors))

    def test_incomplete_case_still_has_actionable_markdown(self):
        markdown = module.render_incomplete_markdown(
            "forward", FileNotFoundError("missing COLMAP metrics")
        )
        self.assertIn("COLMAP golden reference — forward", markdown)
        self.assertIn("Status: FAIL", markdown)
        self.assertIn("missing COLMAP metrics", markdown)


if __name__ == "__main__":
    unittest.main()
