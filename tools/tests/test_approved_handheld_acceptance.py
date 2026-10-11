"""Independent #176 source preparation; no reconstructed geometry oracle."""
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

ROOT = Path(__file__).parents[2]
ORACLE = Path(__file__).parent / 'fixtures/approved-handheld-expectations.json'
spec = importlib.util.spec_from_file_location('corpus_checker', ROOT / 'tools/check_real_video_manifest.py')
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

class HandheldMetadataAcceptance(unittest.TestCase):
    def setUp(self):
        self.expected = json.loads(ORACLE.read_text())

    def test_approved_licensed_pair_metadata_can_be_validated(self):
        self.assertEqual(checker.validate_manifest(self.expected), [])

    def test_repository_contains_both_approved_originals_and_fixed_inventory(self):
        corpus = checker.load_manifest(ROOT / 'benchmarks/real-video-corpus.json')
        by_hash = {c['asset'].get('sha256'): c for c in corpus['cases']}
        for expected in self.expected['cases']:
            actual = by_hash.get(expected['asset']['sha256'])
            self.assertIsNotNone(actual, 'approved handheld original not pinned')
            for field in ['creator','title','license_url','owner_approval_url','download_url','byte_length']:
                self.assertEqual(actual['asset'][field], expected['asset'][field])
            self.assertEqual(actual['preparation']['trim'], expected['preparation']['trim'])
            self.assertEqual(actual['preparation']['inventory'], expected['preparation']['inventory'])

    def test_missing_attribution_integrity_approval_or_inventory_is_rejected(self):
        for section, field in [('asset','creator'),('asset','license_url'),('asset','sha256'),('asset','owner_approval_url'),('preparation','inventory'),('preparation','trim'),('preparation','decoder_boundary')]:
            with self.subTest(section=section, field=field):
                changed = copy.deepcopy(self.expected)
                del changed['cases'][0][section][field]
                self.assertTrue(checker.validate_manifest(changed))
                self.assertNotEqual(checker.validate_manifest(changed), checker.validate_manifest(self.expected), "mutation was not independently rejected")

    def test_false_browser_parity_and_invalid_visibility_are_rejected(self):
        mutations = [
            lambda c: c['preparation']['decoder_boundary'].update(browser_pixel_equivalence=True),
            lambda c: c['preparation']['trim'].update(end_seconds=59),
            lambda c: c['preparation']['inventory']['objects'][0].update(visible_samples=[18]),
            lambda c: c['preparation']['inventory']['objects'][0].update(visible_samples=[]),
            lambda c: c['preparation']['inventory'].update(metric_truth=True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(self.expected)
                mutate(changed['cases'][0])
                self.assertTrue(checker.validate_manifest(changed))
                self.assertNotEqual(checker.validate_manifest(changed), checker.validate_manifest(self.expected), "mutation was not independently rejected")

    def test_unapproved_canary_redistribution_remains_rejected(self):
        corpus = checker.load_manifest(ROOT / 'benchmarks/real-video-corpus.json')
        corpus['cases'][0]['asset']['redistribute'] = True
        self.assertTrue(checker.validate_manifest(corpus))

class ActualPreparedHandheldAcceptance(unittest.TestCase):
    """Explicit real-source tier: run after production preparation, never fake bytes."""
    @classmethod
    def setUpClass(cls):
        source = os.environ.get('VIDEO_HANDHELD_ASSET_ROOT')
        prepared = os.environ.get('VIDEO_HANDHELD_PREPARED_ROOT')
        if not source or not prepared:
            raise unittest.SkipTest('actual licensed sources/prepared outputs are required; product unrun')
        cls.source, cls.prepared = Path(source), Path(prepared)
        cls.expected = json.loads(ORACLE.read_text())['cases']

    def verify(self, case, source, prepared):
        # Existing checker boundary may be translated independently; semantics fixed.
        verify = getattr(checker, 'verify_prepared_case', None)
        self.assertTrue(callable(verify), 'prepared original/derived integrity verification missing')
        return verify(case, source, prepared)

    def test_actual_originals_and_derived_clip_frames_verify(self):
        for case in self.expected:
            original = self.source / case['asset']['expected_file']
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), case['asset']['sha256'])
            self.assertEqual(original.stat().st_size, case['asset']['byte_length'])
            self.assertEqual(self.verify(case, self.source, self.prepared), [])

    def test_actual_receipt_matches_source_plan_and_hashes(self):
        for case in self.expected:
            base = self.prepared / case['id']
            receipt = json.loads((base / 'sampling.json').read_text())
            self.assertEqual(receipt['case_id'], case['id'])
            self.assertEqual(receipt['plan'], case['preparation']['plan'])
            self.assertEqual(receipt['derived']['source_sha256'], case['asset']['sha256'])
            self.assertEqual(receipt['derived']['trim'], case['preparation']['trim'])
            derived = base / receipt['derived']['path']
            self.assertGreater(derived.stat().st_size, 0)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(derived)], capture_output=True, text=True, check=True)
            duration = float(json.loads(probe.stdout)['format']['duration'])
            wanted = case['preparation']['trim']['end_seconds'] - case['preparation']['trim']['start_seconds']
            self.assertLessEqual(abs(duration - wanted), 1 / 30, 'derived clip is not the authorized interval (one source-frame duration allowance)')
            self.assertEqual(hashlib.sha256(derived.read_bytes()).hexdigest(), receipt['derived']['sha256'])
            self.assertEqual(len(receipt['frames']), 18)
            boundary = receipt['decoder_boundary']
            self.assertEqual(boundary['kind'], 'offline-ffmpeg')
            self.assertIs(boundary['browser_pixel_equivalence'], False)
            self.assertTrue(boundary['tool_version'])
            for index, frame in enumerate(receipt['frames']):
                self.assertAlmostEqual(frame['requested_relative_seconds'], receipt['plan']['times'][index], places=6)
                self.assertAlmostEqual(frame['requested_source_seconds'], frame['requested_relative_seconds'] + case['preparation']['trim']['start_seconds'], places=6)
                self.assertIsInstance(frame['selected_source_ordinal'], int)
                self.assertGreaterEqual(frame['selected_source_ordinal'], 0)
                self.assertGreaterEqual(frame['selected_source_pts'], case['preparation']['trim']['start_seconds'])
                self.assertLess(frame['selected_source_pts'], case['preparation']['trim']['end_seconds'])
                path = base / frame['path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), frame['sha256'])
                # Real source-derived PPMs; no generated geometry or stand-in frame.
                self.assertTrue(path.read_bytes().startswith(b'P6'))

    def test_existing_runner_preparation_is_repeatable_without_geometry(self):
        for case in self.expected:
            with tempfile.TemporaryDirectory() as tmp:
                roots = [Path(tmp) / 'first', Path(tmp) / 'second']
                receipts = []
                for output in roots:
                    result = subprocess.run(['bun', 'tools/run_real_video_case.ts', '--case', case['id'], '--asset-root', str(self.source), '--output-root', str(output), '--manifest', str(ORACLE), '--prepare-only'], cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    base = output / case['id']
                    receipt = json.loads((base / 'sampling.json').read_text())
                    self.assertEqual(self.verify(case, self.source, output), [])
                    receipts.append((receipt['derived']['sha256'], [f['sha256'] for f in receipt['frames']]))
                    self.assertEqual(list(base.rglob('*.ply')), [], 'preparation dispatched geometry production')
                    self.assertEqual(list(base.rglob('*.glb')), [], 'preparation dispatched scene/export production')
                self.assertEqual(receipts[0], receipts[1])

    def test_prepared_frame_cannot_escape_its_root(self):
        case = self.expected[0]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copytree(self.prepared / case['id'], tmp / 'prepared' / case['id'])
            base = tmp / 'prepared' / case['id']
            receipt_path = base / 'sampling.json'
            receipt = json.loads(receipt_path.read_text())
            frame = base / receipt['frames'][0]['path']
            outside = tmp / 'outside.ppm'
            shutil.copyfile(frame, outside)
            frame.unlink()
            frame.symlink_to(outside)
            self.assertTrue(self.verify(case, self.source, tmp / 'prepared'), 'outside symlink was accepted with matching bytes')

    def test_wrong_original_derived_or_frame_bytes_fail_closed(self):
        for case in self.expected:
            for target in ['original','derived','frame']:
                with self.subTest(case=case['id'], target=target), tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    shutil.copytree(self.source, tmp / 'source')
                    shutil.copytree(self.prepared / case['id'], tmp / 'prepared' / case['id'])
                    receipt = json.loads((tmp / 'prepared' / case['id'] / 'sampling.json').read_text())
                    if target == 'original':
                        path = tmp / 'source' / case['asset']['expected_file']
                    elif target == 'derived':
                        path = tmp / 'prepared' / case['id'] / receipt['derived']['path']
                        self.assertEqual(receipt['derived']['source_sha256'], case['asset']['sha256'])
                    else:
                        path = tmp / 'prepared' / case['id'] / receipt['frames'][0]['path']
                    with path.open('ab') as handle:
                        handle.write(b'independent tamper')
                    self.assertTrue(self.verify(case, tmp / 'source', tmp / 'prepared'))

if __name__ == '__main__':
    unittest.main()
