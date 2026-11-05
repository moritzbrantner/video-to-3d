"""CPU-only integration coverage for the local MapAnything video wrappers."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "scripts" / "setup_mapanything.sh"
RUN = ROOT / "scripts" / "run_mapanything_video.sh"
PYTHON_RUNNER = ROOT / "scripts" / "run_mapanything_video.py"
spec = importlib.util.spec_from_file_location("run_mapanything_video", PYTHON_RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def synthetic_video(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=80x60:rate=8", "-t", "3", "-c:v", "mpeg4", str(path)],
        check=True, capture_output=True, text=True,
    )


class MapAnythingVideoScriptsTests(unittest.TestCase):
    def test_setup_and_video_entrypoints_explain_usage_without_installing(self):
        for script in (SETUP, RUN):
            with self.subTest(script=script.name):
                process = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True, check=False)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("usage:", process.stdout.lower())

    @unittest.skipUnless(subprocess.run(["bash", "-c", "command -v ffmpeg && command -v ffprobe"],
                                       capture_output=True, check=False).returncode == 0, "FFmpeg unavailable")
    def test_local_video_frames_are_lossless_and_idempotently_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "test clip.mp4"
            synthetic_video(video)
            command = [
                "bash", str(RUN), str(video), "--output-root", str(root / "results"),
                "--view-counts", "8", "--fps", "4", "--max-frames", "12", "--prepare-only",
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            manifests = list((root / "results").glob("*/frames-*/frames.json"))
            self.assertEqual(len(manifests), 1)
            images = sorted(manifests[0].parent.glob("frame-*.png"))
            self.assertEqual(len(images), 12)
            self.assertTrue(images[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            before = [(image.stat().st_mtime_ns, image.stat().st_size) for image in images]
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("Reusing", second.stdout)
            self.assertEqual(before, [(image.stat().st_mtime_ns, image.stat().st_size) for image in images])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_ytdlp_download_is_explicit_and_cached_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "fixture.mp4"
            synthetic_video(video)
            calls = []
            actual_invoke = runner.invoke

            def fake_invoke(command):
                if "yt_dlp" in command:
                    calls.append(command)
                    destination = Path(command[command.index("-o") + 1].replace("%(ext)s", "mp4"))
                    shutil.copyfile(video, destination)
                    return
                actual_invoke(command)

            argv = [
                str(PYTHON_RUNNER), "https://www.youtube.com/watch?v=fixture",
                "--output-root", str(root / "results"), "--fps", "4",
                "--max-frames", "12", "--prepare-only",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {"MAPANYTHING_PYTHON": sys.executable}),
                mock.patch.object(runner, "invoke", side_effect=fake_invoke),
            ):
                self.assertEqual(runner.main(), 0)
                self.assertEqual(runner.main(), 0)
            self.assertEqual(len(calls), 1)
            self.assertIn("--no-playlist", calls[0])
            self.assertEqual(len(list((root / "results").glob("*/frames-*/frames.json"))), 1)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_runner_uses_same_numbered_run_series_only_for_same_frame_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "fixture.mp4"
            synthetic_video(video)
            calls = []
            actual_run = subprocess.run

            def fake_run(command, *args, **kwargs):
                if len(command) > 1 and command[1] == str(runner.SMOKE):
                    calls.append(command)
                    return subprocess.CompletedProcess(command, 0)
                return actual_run(command, *args, **kwargs)

            argv = [str(PYTHON_RUNNER), str(video), "--output-root", str(root / "results"), "--fps", "4"]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {"MAPANYTHING_PYTHON": sys.executable}),
                mock.patch.object(runner.subprocess, "run", side_effect=fake_run),
            ):
                self.assertEqual(runner.main(), 0)
                self.assertEqual(runner.main(), 0)
                sys.argv = argv + ["--fps", "5"]  # Last flag overrides the earlier frame rate.
                self.assertEqual(runner.main(), 0)
            self.assertEqual(len(calls), 3)
            outputs = [command[command.index("--output-dir") + 1] for command in calls]
            self.assertEqual(outputs[0], outputs[1])
            self.assertNotEqual(outputs[0], outputs[2])
            self.assertTrue(all("--revision" in command and "--numbered-runs" in command for command in calls))

    @unittest.skipUnless(subprocess.run(["bash", "-c", "command -v ffmpeg && command -v ffprobe"],
                                       capture_output=True, check=False).returncode == 0, "FFmpeg unavailable")
    def test_insufficient_views_fail_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "short.mp4"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "testsrc2=size=80x60:rate=4", "-t", "1", "-c:v", "mpeg4", str(video)],
                check=True, capture_output=True, text=True,
            )
            process = subprocess.run(
                ["bash", str(RUN), str(video), "--output-root", str(root / "results"),
                 "--fps", "1", "--view-counts", "8", "--prepare-only"],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("need 8", process.stderr)

    def test_dry_run_neither_downloads_nor_creates_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "no-output"
            process = subprocess.run(
                ["bash", str(RUN), "https://www.youtube.com/watch?v=EXAMPLE",
                 "--output-root", str(output), "--dry-run"],
                capture_output=True, text=True, check=False,
                env={**os.environ, "MAPANYTHING_PYTHON": "/does/not/exist"},
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("dry run", process.stdout.lower())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
