import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


class VideoMatchingTests(unittest.TestCase):
    def solve(self, *, budget=None, temporal=3, exhaustive=4, revisits=None, recover=None):
        api = MagicMock()
        source = Path(__file__).resolve().parents[1] / "colmap_golden.py"
        spec = importlib.util.spec_from_file_location("colmap_golden_test", source)
        solver = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, pycolmap=api):
            spec.loader.exec_module(solver)

        def mapped(_database, _images, directory, **_options):
            count = temporal if Path(directory).name == "temporal" else exhaustive
            model = MagicMock()
            model.num_reg_images.return_value = count
            model.num_points3D.return_value = 100
            # Witness which on-disk model is handed off to the next stage.
            (Path(directory) / "witness").write_text(str(count))
            return {0: model}

        api.incremental_mapping.side_effect = mapped
        api.Database.open.return_value.num_images.return_value = 4
        self.diagnostics = {}
        with TemporaryDirectory() as directory, patch.object(solver, "matching_workload", return_value=(4, 11)) as workload:
            work = Path(directory)
            result = solver.solve(
                work, work, "", exhaustive_comparison_budget=budget, propose_revisits=revisits,
                recover_partial=recover, diagnostics=self.diagnostics,
            )
            count = result[0].num_reg_images()
            self.assertEqual((work / "sparse/witness").read_text(), str(count))
        return count, api, workload

    def test_only_a_successful_global_recovery_changes_the_selected_model_and_receipt(self):
        def recover(_db, _images, destination, initial):
            self.assertEqual(initial.num_reg_images(), 3)
            destination.mkdir()
            (destination / "witness").write_text("4")
            result = MagicMock()
            result.num_reg_images.return_value = 4
            return {0: result}

        for candidate, count, mapper in [(recover, 4, "global"), (lambda *args: {}, 3, "incremental")]:
            actual, _, _ = self.solve(exhaustive=3, recover=candidate)
            self.assertEqual(actual, count)
            self.assertEqual(self.diagnostics, {
                "pairing": "exhaustive", "mapper": mapper, "incremental_registered_frames": 3,
            })

    def test_reference_defaults_keep_exhaustive_matching(self):
        count, api, workload = self.solve()
        self.assertEqual(count, 4)
        workload.assert_not_called()
        api.match_exhaustive.assert_called_once()
        api.match_sequential.assert_not_called()

    def test_complete_temporal_model_avoids_all_pairs(self):
        count, api, _ = self.solve(budget=10, temporal=4)
        self.assertEqual(count, 4)
        api.match_sequential.assert_called_once()
        api.match_exhaustive.assert_not_called()

    def test_unregistered_revisit_frames_preserve_the_complete_temporal_model(self):
        pairs = lambda _db, _model: [("000.png", "051.png")]
        count, api, _ = self.solve(budget=10, temporal=4, exhaustive=2, revisits=pairs)
        self.assertEqual(count, 4)
        api.match_image_pairs.assert_called_once()
        api.match_exhaustive.assert_not_called()

    def test_incomplete_temporal_model_reuses_extraction_and_keeps_the_better_result(self):
        for exhaustive, expected in [(4, 4), (2, 3)]:
            with self.subTest(exhaustive=exhaustive):
                count, api, _ = self.solve(budget=10, exhaustive=exhaustive)
                self.assertEqual(count, expected)
                api.extract_features.assert_called_once()
                api.match_sequential.assert_called_once()
                api.match_exhaustive.assert_called_once()


class PartialGlobalRecoveryTests(unittest.TestCase):
    def load_recovery(self):
        source = Path(__file__).resolve().parents[1] / "colmap-sparse" / "global_recovery.py"
        spec = importlib.util.spec_from_file_location("global_recovery_test", source)
        recovery = importlib.util.module_from_spec(spec)
        api = MagicMock()
        with patch.dict(sys.modules, pycolmap=api), patch.object(sys, "path", [str(source.parent), *sys.path]):
            spec.loader.exec_module(recovery)
        return recovery, api

    def test_deregistering_a_weak_view_rechecks_remaining_landmark_support(self):
        recovery, _ = self.load_recovery()
        model = MagicMock()
        model.images = {i: SimpleNamespace(image_id=i, frame_id=i) for i in range(4)}
        model.reg_image_ids.side_effect = lambda: list(model.images)
        pairs = [(0, 1)] * 12 + [(1, 2)] * 7 + [(2, 3)]
        model.points3D = {
            k: SimpleNamespace(track=SimpleNamespace(elements=[model.images[i] for i in pair]))
            for k, pair in enumerate(pairs)
        }

        def drop(frame):
            del model.images[frame]
            for key, point in list(model.points3D.items()):
                if any(e.image_id == frame for e in point.track.elements):
                    del model.points3D[key]

        model.deregister_frame.side_effect = drop
        self.assertTrue(recovery.remove_unsupported_frames(model))
        self.assertEqual(model.reg_image_ids(), [0, 1])

    def test_calibrated_retry_keeps_only_an_expanded_supported_component(self):
        recovery, api = self.load_recovery()

        def model(ids, points, observations):
            result = MagicMock()
            result.images = {i: SimpleNamespace(name=str(i), image_id=i) for i in ids}
            result.reg_image_ids.return_value = ids
            result.num_points3D.return_value = points
            result.compute_num_observations.return_value = observations
            return result

        calibrated = MagicMock()
        initial = model([1, 2, 3], 40, 100)
        initial.cameras = {1: calibrated}
        expanded = model([1, 2, 3, 4], 60, 120)
        api.global_mapping.return_value = {
            0: expanded,
            1: model([1, 2, 5, 6], 70, 150),
            2: model([1, 2, 3, 4], 15, 110),
            3: model([1, 2, 3, 4], 45, 60),
        }
        with TemporaryDirectory() as directory:
            work = Path(directory)
            with patch.object(recovery, "remove_unsupported_frames", return_value=False), patch.object(recovery, "choose_exportable_seed") as choose_seed:
                result = recovery.recover_globally(work / "database", work / "images", work / "recovered", initial)
                choose_seed.side_effect = ValueError("insufficient parallax")
                failed = recovery.recover_globally(work / "database", work / "images", work / "no_seed", initial)
                self.assertFalse(failed)
        self.assertEqual(result, {0: expanded})
        db = api.Database.open.return_value
        self.assertEqual(
            [call.args[0] for call in db.update_camera.call_args_list],
            [calibrated, db.read_camera.return_value] * 2,
        )
