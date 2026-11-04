"""CPU-only contracts for the real-frame MapAnything GPU smoke harness."""

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "smoke_mapanything.py"
spec = importlib.util.spec_from_file_location("smoke_mapanything", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def capture(root: Path, corrupt: bool = False) -> None:
    frame_dir = root / "frames"
    frame_dir.mkdir()
    records = []
    for index in range(4):
        raw = bytes([index, 40, 90, 255] * 6)
        relative = f"frames/{index:03d}.rgba"
        (root / relative).write_bytes(raw)
        records.append(
            {
                "frame_index": index,
                "path": relative,
                "width": 3,
                "height": 2,
                "content_hash": smoke.digest(b"corrupted") if corrupt and index == 2 else smoke.digest(raw),
                "byte_length": len(raw),
                "presented_time": index / 2,
            }
        )
    (root / "browser-frames.json").write_text(json.dumps({"schema_version": 1, "frames": records}), encoding="utf-8")


class MapAnythingSmokeTests(unittest.TestCase):
    def test_sequential_or_uniform_view_sampling_is_deterministic(self):
        self.assertEqual(smoke.select_indices(18, 2), [0, 1])
        self.assertEqual(smoke.select_indices(18, 4, start=3), [3, 4, 5, 6])
        self.assertEqual(smoke.select_indices(18, 2, policy="uniform"), [0, 17])
        self.assertEqual(smoke.select_indices(18, 4, policy="uniform"), [0, 5, 11, 17])
        self.assertEqual(smoke.select_indices(4, 4), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            smoke.select_indices(4, 5)

    def test_png_preserves_rgb_without_new_video_decoder(self):
        rgba = bytes([10, 20, 30, 255, 40, 50, 60, 255])
        png = smoke.rgba_to_png(rgba, 2, 1)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II", png[16:24]), (2, 1))
        # PNG IDAT: the leading zero is the unfiltered scanline marker.
        length = struct.unpack(">I", png[33:37])[0]
        self.assertEqual(zlib.decompress(png[41 : 41 + length]), bytes([0, 10, 20, 30, 40, 50, 60]))
        with self.assertRaisesRegex(ValueError, "transparency"):
            smoke.rgba_to_png(bytes([10, 20, 30, 0]), 1, 1)

    def test_browser_capture_receipts_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capture"
            root.mkdir()
            capture(root)
            images, sources = smoke.read_capture(root, Path(tmp) / "converted")
            self.assertEqual(len(images), 4)
            self.assertEqual(sources[2]["source_sha256"], smoke.digest(bytes([2, 40, 90, 255] * 6)))
            self.assertTrue(all(path.is_file() for path in images))

    def test_corrupted_browser_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capture"
            root.mkdir()
            capture(root, corrupt=True)
            with self.assertRaisesRegex(ValueError, "receipt"):
                smoke.read_capture(root, Path(tmp) / "converted")

    def test_capture_rejects_paths_outside_the_authoritative_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capture"
            root.mkdir()
            capture(root)
            manifest = root / "browser-frames.json"
            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["frames"][0]["path"] = "../another-file.rgba"
            (Path(tmp) / "another-file.rgba").write_bytes(bytes([0, 40, 90, 255] * 6))
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                smoke.read_capture(root, Path(tmp) / "converted")

    def test_only_immutable_completed_evidence_can_be_reused(self):
        revision = "a" * 40
        plan = {"cases": [{"views": 2}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = {
                "plan": plan,
                "checkpoint_commit": revision,
                "cases": [{"status": "model_completed"}],
                "status": "completed",
            }
            smoke.write_json(root / "summary.json", summary)
            (root / "share-logs.zip").write_bytes(b"archive")
            self.assertTrue(smoke.can_reuse_report(root, plan, revision))
            self.assertFalse(smoke.can_reuse_report(root, plan, "main"))
            self.assertFalse(smoke.can_reuse_report(root, plan, "b" * 40))
            summary["status"] = "partial_or_failed"
            smoke.write_json(root / "summary.json", summary)
            self.assertFalse(smoke.can_reuse_report(root, plan, revision))

    def test_missing_nvidia_gpu_stops_before_model_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capture"
            root.mkdir()
            capture(root)
            output = Path(tmp) / "results"
            argv = [str(SCRIPT), "--capture-root", str(root), "--output-dir", str(output), "--view-counts", "2,4"]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(smoke, "hardware", return_value={"nvidia_smi": "unavailable"}),
                mock.patch.object(smoke, "child") as child,
            ):
                self.assertEqual(smoke.main(), 1)
            child.assert_not_called()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "gpu_unavailable")
            with smoke.zipfile.ZipFile(output / "share-logs.zip") as archive:
                self.assertIn("summary.json", archive.namelist())

    def test_separate_child_collects_sampled_device_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(smoke, "device_used_mib", side_effect=[101, 165]):
                code, elapsed, metrics = smoke.child(
                    [sys.executable, "-c", "print('ok')"], Path(tmp), 10, dict(smoke.os.environ), measure_gpu=True
                )
            self.assertEqual(code, 0)
            self.assertGreaterEqual(elapsed, 0)
            self.assertEqual(metrics["nvidia_smi_gpu0_baseline_used_mib"], 101)
            self.assertEqual(metrics["nvidia_smi_gpu0_peak_used_sampled_mib"], 165)
            self.assertEqual(metrics["nvidia_smi_gpu0_peak_increase_sampled_mib"], 64)
            self.assertIn("ok", (Path(tmp) / "stdout.log").read_text(encoding="utf-8"))

    def test_interrupted_checkpoint_resolution_keeps_shareable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capture"
            root.mkdir()
            capture(root)
            output = Path(tmp) / "results"
            argv = [str(SCRIPT), "--capture-root", str(root), "--output-dir", str(output), "--view-counts", "2,4"]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(smoke, "hardware", return_value={"nvidia_smi": "RTX 3060 Ti, 8192, 580"}),
                mock.patch.object(smoke, "child", side_effect=KeyboardInterrupt),
                mock.patch.dict(smoke.os.environ, {"XDG_CACHE_HOME": str(Path(tmp) / "cache")}),
                mock.patch.object(smoke, "git_head", return_value="a" * 40),
            ):
                self.assertEqual(smoke.main(), 130)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "interrupted_during_checkpoint_resolution")
            self.assertTrue((output / "share-logs.zip").is_file())

    def test_dry_run_needs_no_gpu_or_model_and_has_no_output_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capture"
            root.mkdir()
            capture(root)
            output = Path(tmp) / "results"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--capture-root", str(root), "--view-counts", "2,4", "--output-dir", str(output), "--dry-run"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(
                report["cases"],
                [{"views": 2, "source_indices": [0, 1]}, {"views": 4, "source_indices": [0, 1, 2, 3]}],
            )
            self.assertFalse(report["upstream_geometry_accepted_by_rust"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
