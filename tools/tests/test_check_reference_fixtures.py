import copy
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


def manifest():
    return {
        "schema_version": 2,
        "stage_order": ["seed", "registration", "mesh"],
        "stages": {
            "seed": {"default_checks": {"seed_selected": {"equals": True}}},
            "registration": {"default_checks": {"registered_images": {"min": 9}}},
            "mesh": {
                "default_checks": {
                    "median_surface_error_m": {"max": 0.03},
                    "cross_reference_triangles": {"min": 1},
                }
            },
        },
        "fixtures": [overlap_fixture()],
    }


def overlap_fixture(expected_failures=None):
    return {
        "id": "overlapping-references",
        "intent": "one surface",
        "colmap_expectation": "registers",
        "checks": {"seed": "default", "registration": "default", "mesh": "default"},
        "baseline": {
            "recorded": "2026-10-09",
            "first_failing_stage": "mesh",
            "expected_failures": expected_failures
            if expected_failures is not None
            else [
                {"stage": "mesh", "check": "cross_reference_triangles", "issue": 126, "reason": "no spans"}
            ],
        },
    }


def stages(**overrides):
    report = {
        "seed": {"available": True, "metrics": {"seed_selected": True}},
        "registration": {"available": True, "metrics": {"registered_images": 10}},
        "mesh": {
            "available": True,
            "metrics": {"median_surface_error_m": 0.01, "cross_reference_triangles": 0},
        },
    }
    for key, value in overrides.items():
        stage, metric = key.split("__")
        report[stage]["metrics"][metric] = value
    return report


class ReferenceFixtureGateTests(unittest.TestCase):
    def evaluate(self, fixture, report):
        definition = manifest()
        definition["fixtures"] = [fixture]
        return module.evaluate(definition, fixture, report)

    def test_recorded_failures_are_expected(self):
        result = self.evaluate(overlap_fixture(), stages())
        self.assertEqual((result["status"], result["errors"]), ("EXPECTED FAILURE", []))
        self.assertEqual(result["first_failing_stage"], "mesh")

    def test_new_failure_is_fatal_and_names_the_first_failing_stage(self):
        result = self.evaluate(overlap_fixture(), stages(registration__registered_images=4))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["first_failing_stage"], "registration")
        self.assertIn("registration.registered_images", result["errors"][0])

    def test_recovered_failure_demands_a_baseline_update(self):
        result = self.evaluate(overlap_fixture(), stages(mesh__cross_reference_triangles=40))
        self.assertEqual(result["status"], "UNEXPECTED PASS")
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("#126", result["errors"][0])

    def test_all_checks_met_without_expected_failures_passes(self):
        result = self.evaluate(overlap_fixture([]), stages(mesh__cross_reference_triangles=40))
        self.assertEqual((result["status"], result["errors"]), ("PASS", []))
        self.assertIsNone(result["first_failing_stage"])

    def test_missing_or_non_finite_metric_fails_its_check(self):
        for value in (None, float("nan")):
            result = self.evaluate(overlap_fixture(), stages(mesh__median_surface_error_m=value))
            self.assertEqual(result["status"], "FAIL")

    def test_unavailable_stage_fails_every_check_as_not_reached(self):
        report = stages()
        report["mesh"] = {"available": False, "blocked_by": "no registered cameras"}
        result = self.evaluate(overlap_fixture(), report)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("not reached (no registered cameras)", result["errors"][0])

    def test_stage_level_expected_failure_covers_every_check(self):
        fixture = overlap_fixture([{"stage": "mesh", "issue": 127, "reason": "blocked"}])
        report = stages()
        report["mesh"] = {"available": False, "blocked_by": "no registered cameras"}
        self.assertEqual(self.evaluate(fixture, report)["status"], "EXPECTED FAILURE")
        recovered = stages(mesh__cross_reference_triangles=3)
        self.assertEqual(self.evaluate(fixture, recovered)["status"], "UNEXPECTED PASS")

    def test_negative_control_requires_the_baseline_gate(self):
        fixture = overlap_fixture([])
        fixture["id"] = "pure-rotation"
        fixture["checks"] = {
            "seed": {"seed_selected": {"equals": False}, "decisive_gate": {"equals": "baseline"}},
            "registration": {"registered_images": {"max": 0}},
        }
        accepted = stages(seed__seed_selected=False, seed__decisive_gate="baseline", registration__registered_images=0)
        self.assertEqual(self.evaluate(fixture, accepted)["status"], "PASS")
        wrong_gate = copy.deepcopy(accepted)
        wrong_gate["seed"]["metrics"]["decisive_gate"] = "parallax"
        self.assertEqual(self.evaluate(fixture, wrong_gate)["status"], "FAIL")
        registered = copy.deepcopy(accepted)
        registered["registration"]["metrics"]["registered_images"] = 4
        self.assertEqual(self.evaluate(fixture, registered)["status"], "FAIL")

    def test_boolean_is_not_a_number(self):
        fixture = overlap_fixture([])
        self.assertFalse(module.check_result({"min": 1}, True))
        self.assertTrue(module.check_result({"equals": True}, True))
        self.assertEqual(self.evaluate(fixture, stages(registration__registered_images=True))["status"], "FAIL")

    def test_repository_manifest_is_valid(self):
        definition = module.load_manifest(module.DEFAULT_MANIFEST)
        self.assertEqual(
            [fixture["id"] for fixture in definition["fixtures"]],
            ["slow-lateral-pan", "pure-rotation", "overlapping-references"],
        )
        self.assertEqual(
            definition["stage_order"],
            ["sampling", "features", "seed", "registration", "dense", "mesh", "texture"],
        )

    def write_manifest(self, definition):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "manifest.json"
        path.write_text(json.dumps(definition))
        return path

    def test_manifest_rejects_expected_failure_outside_the_checks(self):
        definition = manifest()
        definition["fixtures"] = [
            overlap_fixture([{"stage": "mesh", "check": "shared_vertices", "issue": 1, "reason": "x"}])
        ]
        with self.assertRaises(ValueError):
            module.load_manifest(self.write_manifest(definition))

    def test_manifest_rejects_expected_failure_without_issue(self):
        definition = manifest()
        definition["fixtures"] = [overlap_fixture([{"stage": "mesh", "reason": "x"}])]
        with self.assertRaises(ValueError):
            module.load_manifest(self.write_manifest(definition))

    def test_manifest_rejects_ambiguous_bounds(self):
        definition = manifest()
        definition["stages"]["seed"]["default_checks"] = {"seed_selected": {"min": 1, "max": 2}}
        with self.assertRaises(ValueError):
            module.load_manifest(self.write_manifest(definition))

    def test_stage_report_must_match_the_case(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_dir = Path(directory)
            (fixture_dir / "stage-report.json").write_text(
                json.dumps({"schema": module.STAGE_REPORT_SCHEMA, "case": "slow-lateral-pan", "stages": {}})
            )
            with self.assertRaises(ValueError):
                module.load_stages(fixture_dir, "overlapping-references")
            loaded = module.load_stages(fixture_dir, "slow-lateral-pan")
            self.assertFalse(loaded["sampling"]["available"])

    def test_markdown_names_first_failing_stage_and_reports_colmap_without_gating(self):
        fixture = overlap_fixture()
        result = self.evaluate(fixture, stages())
        markdown = module.render_markdown(fixture, result, {"registered_images": "0"})
        self.assertIn("COLMAP reference: no model", markdown)
        self.assertIn("First failing stage: **mesh**", markdown)
        self.assertIn("expected failure (#126)", markdown)
        moved = self.evaluate(fixture, stages(registration__registered_images=2))
        self.assertIn("(baseline: mesh)", module.render_markdown(fixture, moved, None))


if __name__ == "__main__":
    unittest.main()
