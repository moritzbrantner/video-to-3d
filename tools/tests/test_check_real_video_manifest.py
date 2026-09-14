from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "check_real_video_manifest.py"
spec = importlib.util.spec_from_file_location("check_real_video_manifest", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RealVideoManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        manifest_path = Path(__file__).parents[2] / "benchmarks" / "real-video-corpus.json"
        self.manifest = module.load_manifest(manifest_path)

    def test_repository_manifest_is_valid(self) -> None:
        self.assertEqual(module.validate_manifest(self.manifest), [])

    def test_duplicate_case_ids_are_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][1]["id"] = manifest["cases"][0]["id"]
        self.assertTrue(
            any("duplicate case id" in error for error in module.validate_manifest(manifest))
        )

    def test_manual_assets_cannot_be_scheduled(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][2]["automation"]["scheduled"] = True
        self.assertTrue(
            any("manual assets cannot be scheduled" in error for error in module.validate_manifest(manifest))
        )

    def test_asset_paths_cannot_escape_data_root(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][0]["asset"]["expected_file"] = "../Ignatius.mp4"
        self.assertTrue(
            any("safe relative path" in error for error in module.validate_manifest(manifest))
        )

    def test_sources_must_use_https(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][0]["asset"]["source_url"] = "http://example.invalid/video.mp4"
        self.assertTrue(any("source_url must be https" in error for error in module.validate_manifest(manifest)))


if __name__ == "__main__":
    unittest.main()
