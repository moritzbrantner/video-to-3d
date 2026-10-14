"""Independent #187 source preparation acceptance; no inferred scene/Blender pass."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from decimal import Decimal

ROOT = Path(__file__).parents[2]
ORACLE = Path(__file__).parent / 'fixtures/approved-controlled-room-expectations.json'
REVIEW = Path(__file__).parent / 'fixtures/approved-controlled-room-source-review.json'
spec = importlib.util.spec_from_file_location('independent_source_reference', Path(__file__).with_name('test_approved_handheld_acceptance.py'))
assert spec and spec.loader
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
checker = reference.checker


def document():
    return json.loads(ORACLE.read_text())


def files(root):
    return {str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


class ControlledRoomMetadataAcceptance(unittest.TestCase):
    def test_exact_approved_whole_rgb_movie_and_nonmetric_inventory_are_supported(self):
        expected = document()
        self.assertEqual(checker.validate_manifest(expected), [])
        case = expected['cases'][0]
        self.assertEqual(case['preparation']['trim'], {'start_seconds': 0, 'end_seconds': 86.166667})
        inventory = case['preparation']['inventory']
        self.assertIs(inventory['metric_truth'], False)
        self.assertEqual(inventory['minimum_recognizable_editable_objects'], 3)
        self.assertGreaterEqual(len(inventory['objects']), 3)

    def test_repository_pins_approved_source_fixed_sampling_and_independent_inventory(self):
        expected = document()['cases'][0]
        corpus = checker.load_manifest(ROOT / 'benchmarks/real-video-corpus.json')
        matches = [case for case in corpus['cases'] if case.get('id') == expected['id']]
        self.assertEqual(len(matches), 1, 'approved controlled-room preparation is missing')
        for field in ('creator', 'title', 'citation', 'source_url', 'download_url', 'license_url',
                      'owner_approval_url', 'sha256', 'byte_length', 'expected_file'):
            self.assertEqual(matches[0]['asset'][field], expected['asset'][field])
        self.assertEqual(matches[0]['preparation'], expected['preparation'])

    def test_mutated_approval_or_whole_loop_inventory_cannot_inherit_pinned_identity(self):
        expected = document()
        self.assertEqual(checker.validate_manifest(expected), [], 'valid source is a precondition')
        mutations = [
            lambda case: case['asset'].update(license_url='https://creativecommons.org/licenses/by/3.0/'),
            lambda case: case['asset'].update(sha256='0' * 64),
            lambda case: case['asset'].pop('creator'),
            lambda case: case['preparation']['trim'].update(start_seconds=1),
            lambda case: case['preparation']['plan']['times'].__setitem__(0, 5),
            lambda case: case['preparation']['decoder_boundary'].update(browser_pixel_equivalence=True),
            lambda case: case['preparation']['inventory'].update(metric_truth=True),
            lambda case: case['preparation']['inventory']['objects'][0].update(visible_samples=[18]),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(expected)
            mutate(changed['cases'][0])
            with self.subTest(changed=changed['cases'][0]):
                self.assertTrue(checker.validate_manifest(changed))


class ControlledRoomActualAcceptance(unittest.TestCase):
    def source(self):
        value = os.environ.get('VIDEO_CONTROLLED_ASSET_ROOT')
        if not value:
            self.skipTest('licensed original required; actual controlled-room correspondence UNRUN')
        return Path(value)

    def test_original_nearest_pts_and_pixels_match_independent_fixed_review(self):
        case = document()['cases'][0]
        original = self.source() / case['asset']['expected_file']
        decoded = reference.decoded_source_reference(str(original), json.dumps(case))
        fixed = json.loads(REVIEW.read_text())
        self.assertEqual(fixed['source_sha256'], case['asset']['sha256'])
        self.assertEqual(len(decoded), 18)
        for (ordinal, pts, ppm), sample in zip(decoded, fixed['samples']):
            self.assertEqual(ordinal, sample['selected_source_ordinal'])
            self.assertEqual(pts, Decimal(str(sample['selected_source_pts'])))
            self.assertEqual(hashlib.sha256(ppm).hexdigest(), sample['ppm_sha256'])
            self.assertEqual(reference.p6_pixels(ppm)[:2], (360, 270))

    def test_actual_public_preparation_is_portable_repeatable_and_rejects_self_consistent_tampering(self):
        source = self.source()
        case = document()['cases'][0]
        original = source / case['asset']['expected_file']
        source_before = (hashlib.sha256(original.read_bytes()).hexdigest(), original.stat().st_mtime_ns)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'prepared'
            command = ['bun', 'tools/run_real_video_case.ts', '--case', case['id'], '--asset-root', str(source),
                       '--output-root', str(output), '--manifest', str(ORACLE), '--prepare-only']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            base = output / case['id']
            receipt = json.loads((base / 'sampling.json').read_text())
            self.assertEqual(checker.verify_prepared_case(case, source, output), [])
            self.assertEqual(receipt['original']['sha256'], case['asset']['sha256'])
            self.assertEqual(receipt['derived']['source_sha256'], case['asset']['sha256'])
            self.assertEqual(receipt['derived']['trim'], case['preparation']['trim'])
            self.assertIs(receipt['decoder_boundary']['browser_pixel_equivalence'], False)
            self.assertEqual(len(receipt['frames']), 18)
            decoded = reference.decoded_source_reference(str(original), json.dumps(case))
            for actual, (ordinal, pts, ppm) in zip(receipt['frames'], decoded):
                self.assertEqual(actual['selected_source_ordinal'], ordinal)
                self.assertEqual(Decimal(str(actual['selected_source_pts'])), pts)
                self.assertEqual(reference.p6_pixels((base / actual['path']).read_bytes()), reference.p6_pixels(ppm))
            self.assertEqual(list(base.rglob('*.ply')), [])
            self.assertEqual(list(base.rglob('*.glb')), [])
            before = files(output)
            repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(repeat.returncode, 0, repeat.stdout + repeat.stderr)
            self.assertEqual(files(output), before, 'repeat preparation must not rewrite verified outputs')
            self.assertEqual((hashlib.sha256(original.read_bytes()).hexdigest(), original.stat().st_mtime_ns), source_before)
            # Correct hashes alone cannot bless a different source frame and PTS.
            changed = Path(temporary) / 'tampered'
            shutil.copytree(output, changed)
            altered_base = changed / case['id']
            altered = copy.deepcopy(receipt)
            first, second = altered['frames'][:2]
            payload = (altered_base / second['path']).read_bytes()
            (altered_base / first['path']).write_bytes(payload)
            first.update(sha256=hashlib.sha256(payload).hexdigest(),
                         selected_source_ordinal=second['selected_source_ordinal'], selected_source_pts=second['selected_source_pts'])
            (altered_base / 'sampling.json').write_text(json.dumps(altered))
            self.assertTrue(checker.verify_prepared_case(case, source, changed))

    def test_public_prepare_rejects_unauthorized_trim_before_writing_outputs(self):
        source = self.source()
        expected = document()
        self.assertEqual(checker.validate_manifest(expected), [], 'valid source is a precondition')
        expected['cases'][0]['preparation']['trim']['start_seconds'] = 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'unapproved.json'
            manifest.write_text(json.dumps(expected))
            output = root / 'prepared'
            output.mkdir()
            sentinel = output / 'retained.txt'
            sentinel.write_text('retain')
            before = files(output)
            result = subprocess.run(['bun', 'tools/run_real_video_case.ts', '--case', expected['cases'][0]['id'],
                                     '--asset-root', str(source), '--output-root', str(output),
                                     '--manifest', str(manifest), '--prepare-only'], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue('trim' in (result.stdout + result.stderr).lower() or 'interval' in (result.stdout + result.stderr).lower())
            self.assertEqual(files(output), before)


if __name__ == '__main__':
    unittest.main()
