import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "check_reference_fixtures.py"
spec = importlib.util.spec_from_file_location("check_reference_fixtures", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def overlap_fixture(expected_failures=None):
    return {
        "id": "overlapping-references",
        "intent": "one surface",
        "target": {
            "min_registered_images": 9,
            "mesh_components": 1,
            "min_cross_reference_triangles": 1,
        },
        "colmap_expectation": "registers",
        "baseline": {
            "recorded": "2026-10-09",
            "expected_failures": expected_failures
            if expected_failures is not None
            else [
                {"check": "mesh_components", "issue": 126, "reason": "separate sheets"},
                {"check": "min_cross_reference_triangles", "issue": 126, "reason": "no spans"},
            ],
            "rust": {"registered_images": 10},
        },
    }


def rust_metrics(**overrides):
    metrics = {
        "case": "overlapping-references",
        "registered_images": "10",
        "normalized_pose_rmse": "0.006",
        "decisive_gate": "none",
        "mesh_components": "103",
        "cross_reference_triangles": "0",
    }
    metrics.update({key: str(value) for key, value in overrides.items()})
    return metrics


class ReferenceFixtureGateTests(unittest.TestCase):
    def test_recorded_failures_are_expected(self):
        status, errors = module.evaluate(overlap_fixture(), rust_metrics())
        self.assertEqual(status, "EXPECTED FAILURE")
        self.assertEqual(errors, [])

    def test_new_failure_is_fatal(self):
        status, errors = module.evaluate(overlap_fixture(), rust_metrics(registered_images=4))
        self.assertEqual(status, "FAIL")
        self.assertIn("min_registered_images", errors[0])

    def test_recovered_failure_demands_a_baseline_update(self):
        status, errors = module.evaluate(
            overlap_fixture(), rust_metrics(mesh_components=1, cross_reference_triangles=40)
        )
        self.assertEqual(status, "UNEXPECTED PASS")
        self.assertEqual(len(errors), 2)
        self.assertIn("#126", errors[0])

    def test_all_targets_met_without_expected_failures_passes(self):
        status, errors = module.evaluate(
            overlap_fixture(expected_failures=[]),
            rust_metrics(mesh_components=1, cross_reference_triangles=40),
        )
        self.assertEqual((status, errors), ("PASS", []))

    def test_non_finite_metric_fails_its_check(self):
        fixture = overlap_fixture(expected_failures=[])
        fixture["target"] = {"max_normalized_pose_rmse": 0.25}
        status, _ = module.evaluate(fixture, rust_metrics(normalized_pose_rmse="nan"))
        self.assertEqual(status, "FAIL")

    def test_negative_control_requires_the_baseline_gate(self):
        fixture = overlap_fixture(expected_failures=[])
        fixture["id"] = "pure-rotation"
        fixture["target"] = {"max_registered_images": 0, "decisive_gate": "baseline"}
        accepted = rust_metrics(case="pure-rotation", registered_images=0, decisive_gate="baseline")
        self.assertEqual(module.evaluate(fixture, accepted)[0], "PASS")
        wrong_gate = rust_metrics(case="pure-rotation", registered_images=0, decisive_gate="parallax")
        self.assertEqual(module.evaluate(fixture, wrong_gate)[0], "FAIL")
        registered = rust_metrics(case="pure-rotation", registered_images=4, decisive_gate="none")
        self.assertEqual(module.evaluate(fixture, registered)[0], "FAIL")

    def test_case_mismatch_fails(self):
        status, _ = module.evaluate(overlap_fixture(), rust_metrics(case="slow-lateral-pan"))
        self.assertEqual(status, "FAIL")

    def test_repository_manifest_is_valid(self):
        manifest = module.load_manifest(module.DEFAULT_MANIFEST)
        self.assertEqual(
            [fixture["id"] for fixture in manifest["fixtures"]],
            ["slow-lateral-pan", "pure-rotation", "overlapping-references"],
        )

    def test_manifest_rejects_expected_failure_outside_target(self):
        manifest = {
            "schema_version": 1,
            "fixtures": [
                overlap_fixture(
                    expected_failures=[{"check": "max_registered_images", "issue": 1, "reason": "x"}]
                )
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                module.load_manifest(path)

    def test_markdown_reports_colmap_without_gating(self):
        markdown = module.render_markdown(
            overlap_fixture(),
            rust_metrics(),
            {"registered_images": "0"},
            "EXPECTED FAILURE",
            [],
        )
        self.assertIn("COLMAP reference: no model", markdown)
        self.assertIn("Expected failure `mesh_components` (#126)", markdown)


if __name__ == "__main__":
    unittest.main()
