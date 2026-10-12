"""Independent #176 source preparation; no reconstructed geometry oracle."""
from __future__ import annotations
import copy
from decimal import Decimal
from functools import lru_cache
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

REVIEW = Path(__file__).parent / 'fixtures/approved-handheld-source-review.json'

def nearest_original_frame(pts, requested, start, end):
    """Independent linear reference: exact decimal distance, earlier PTS/ordinal ties."""
    candidates = [n for n, value in enumerate(pts) if start <= value < end]
    if not candidates:
        raise ValueError('authorized interval has no decoded frame')
    return min(candidates, key=lambda n: (abs(pts[n] - requested), pts[n], n))

@lru_cache(maxsize=2)
def decoded_source_reference(original_text, case_json):
    # Explicit offline acceptance oracle; never imports the producer's selection code.
    original = Path(original_text)
    case = json.loads(case_json)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == case['asset']['sha256']
    assert original.stat().st_size == case['asset']['byte_length']
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time',
        '-of', 'json', str(original)], capture_output=True, text=True, check=True)
    pts = [Decimal(frame['best_effort_timestamp_time'])
           for frame in json.loads(probe.stdout)['frames']]
    trim, plan = case['preparation']['trim'], case['preparation']['plan']
    start, end = Decimal(str(trim['start_seconds'])), Decimal(str(trim['end_seconds']))
    ordinals = [nearest_original_frame(pts, Decimal(str(t)) + start, start, end)
                for t in plan['times']]
    selection = '+'.join(f'eq(n,{n})' for n in ordinals)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(original),
            '-vf', f"select='{selection}',scale={plan['analysisWidth']}:{plan['analysisHeight']}:flags=bilinear",
            '-fps_mode', 'passthrough', str(output / 'frame-%02d.ppm')], check=True)
        frames = [(output / f'frame-{i+1:02d}.ppm').read_bytes() for i in range(len(ordinals))]
    return [(ordinal, pts[ordinal], frame) for ordinal, frame in zip(ordinals, frames)]

def p6_pixels(data):
    # Header formatting is not pixel identity. Support comments and exact binary length.
    position, tokens = 0, []
    while len(tokens) < 4:
        while data[position:position+1].isspace():
            position += 1
        if data[position:position+1] == b'#':
            position = data.index(b'\n', position) + 1
            continue
        end = position
        while end < len(data) and not data[end:end+1].isspace():
            end += 1
        tokens.append(data[position:end])
        position = end
    if tokens[0] != b'P6' or tokens[3] != b'255' or not data[position:position+1].isspace():
        raise ValueError('expected 8-bit binary P6')
    position += 2 if data[position:position+2] == b'\r\n' else 1
    width, height = int(tokens[1]), int(tokens[2])
    pixels = data[position:]
    if len(pixels) != width * height * 3:
        raise ValueError('P6 pixel length does not match dimensions')
    return width, height, pixels

class SourceSelectionAcceptance(unittest.TestCase):
    def test_nearest_selection_has_earlier_tie_and_excludes_trim_end(self):
        pts = list(map(Decimal, ['0', '0.2', '0.4', '0.6']))
        self.assertEqual(nearest_original_frame(pts, Decimal('0.3'), Decimal('0.2'), Decimal('0.6')), 1)
        self.assertEqual(nearest_original_frame(pts, Decimal('0.39'), Decimal('0.2'), Decimal('0.6')), 2)
        self.assertEqual(nearest_original_frame(pts, Decimal('0.59'), Decimal('0.2'), Decimal('0.6')), 2)

    def test_actual_original_selection_and_pixels_reproduce_fixed_source_review(self):
        source = os.environ.get('VIDEO_HANDHELD_ASSET_ROOT')
        if not source:
            self.skipTest('licensed original sources required; offline correspondence unrun')
        review = json.loads(REVIEW.read_text())['cases']
        for case in json.loads(ORACLE.read_text())['cases']:
            reference = decoded_source_reference(str(Path(source) / case['asset']['expected_file']), json.dumps(case))
            self.assertEqual(len(reference), 18)
            self.assertEqual(len(review[case['id']]['samples']), 18)
            for (ordinal, pts, ppm), fixed in zip(reference, review[case['id']]['samples']):
                self.assertEqual(ordinal, fixed['selected_source_ordinal'])
                self.assertEqual(pts, Decimal(str(fixed['selected_source_pts'])))
                self.assertEqual(hashlib.sha256(ppm).hexdigest(), fixed['ppm_sha256'])
                self.assertEqual(p6_pixels(ppm)[:2], (360, 270))

def unapproved_trim_cases(expected):
    changed = copy.deepcopy(expected)
    changed['cases'][0]['preparation']['trim'] = {'start_seconds': 59, 'end_seconds': 85}
    h3 = changed['cases'][1]
    h3['preparation']['trim'] = {'start_seconds': 0, 'end_seconds': 24.3}
    # Ask the actual public planner: a plan mismatch must not fake approval rejection.
    planned = subprocess.run(['bun', '-e',
        'import {buildVideoSamplingPlan} from "./apps/web/src/videoSampling"; console.log(JSON.stringify(buildVideoSamplingPlan({duration:24.3,videoWidth:640,videoHeight:480},{framesPerSecond:1.25,maxFrames:18})))'],
        cwd=ROOT, capture_output=True, text=True, check=True)
    h3['preparation']['plan'] = json.loads(planned.stdout)
    return changed

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

    def test_unapproved_trim_cannot_inherit_fixed_source_approval(self):
        changed = unapproved_trim_cases(self.expected)
        for case in changed['cases']:
            with self.subTest(case=case['id']):
                manifest = copy.deepcopy(self.expected)
                manifest['cases'] = [case]
                errors = checker.validate_manifest(manifest)
                self.assertTrue(errors, 'unapproved interval inherited the source approval')
                self.assertTrue(any('trim' in error.lower() or 'interval' in error.lower() for error in errors), errors)

    def test_alternate_plan_or_duration_cannot_inherit_fixed_source_approval(self):
        for field in ['plan', 'original_duration_seconds']:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.expected)
                preparation = changed['cases'][0]['preparation']
                if field == 'plan':
                    preparation['plan']['times'][0] += 0.001
                else:
                    preparation[field] += 0.1
                errors = checker.validate_manifest(changed)
                self.assertTrue(errors, 'alternate source preparation inherited fixed approval')
                self.assertTrue(any(field in error.lower() or 'approved' in error.lower() for error in errors), errors)

    def test_unapproved_canary_redistribution_remains_rejected(self):
        corpus = checker.load_manifest(ROOT / 'benchmarks/real-video-corpus.json')
        corpus['cases'][0]['asset']['redistribute'] = True
        self.assertTrue(checker.validate_manifest(corpus))

class ActualUnapprovedTrimAcceptance(unittest.TestCase):
    def test_existing_prepare_only_rejects_unapproved_trim_before_writes(self):
        source = os.environ.get('VIDEO_HANDHELD_ASSET_ROOT')
        if not source:
            self.skipTest('licensed original sources required; public prepare rejection unrun')
        changed = unapproved_trim_cases(json.loads(ORACLE.read_text()))
        for case in changed['cases']:
            with self.subTest(case=case['id']), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                output = base / 'outputs'
                output.mkdir()
                sentinel = output / 'retained.txt'
                sentinel.write_text('unchanged independent evidence')
                manifest = base / 'unapproved.json'
                document = copy.deepcopy(changed)
                document['cases'] = [case]
                manifest.write_text(json.dumps(document))
                result = subprocess.run(['bun', 'tools/run_real_video_case.ts', '--case', case['id'],
                    '--asset-root', source, '--output-root', str(output), '--manifest', str(manifest),
                    '--prepare-only'], cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, 'public preparation encoded an unapproved source interval; actual outputs=' + str(sorted(str(path.relative_to(output)) for path in output.rglob('*') if path.is_file())) + '; stdout=' + result.stdout)
                reason = (result.stdout + result.stderr).lower()
                self.assertTrue('trim' in reason or 'interval' in reason, 'failure did not explain authorization boundary: ' + reason)
                self.assertEqual(sentinel.read_text(), 'unchanged independent evidence')
                self.assertEqual(sorted(path.name for path in output.iterdir()), ['retained.txt'], 'rejected preparation wrote output')

    def test_direct_python_preparation_rejects_alternate_approved_plan_before_writes(self):
        source = os.environ.get('VIDEO_HANDHELD_ASSET_ROOT')
        if not source:
            self.skipTest('licensed original sources required; direct preparation rejection unrun')
        document = json.loads(ORACLE.read_text())
        document['cases'] = [document['cases'][0]]
        case = document['cases'][0]
        case['preparation']['plan']['times'][0] += 0.001
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / 'outputs'
            output.mkdir()
            manifest = base / 'unapproved-plan.json'
            manifest.write_text(json.dumps(document))
            result = subprocess.run(['python3', 'tools/check_real_video_manifest.py', str(manifest),
                '--prepare-case', case['id'], '--asset-root', source, '--output-root', str(output)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, 'direct preparation accepted alternative fixed-plan request; actual outputs=' + str(sorted(str(path.relative_to(output)) for path in output.rglob('*') if path.is_file())) + '; stdout=' + result.stdout)
            reason = (result.stdout + result.stderr).lower()
            self.assertTrue('plan' in reason or 'approved' in reason, 'unrelated failure: ' + reason)
            self.assertEqual(list(output.iterdir()), [], 'rejected direct preparation wrote output')

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

    def test_actual_browser_can_decode_both_approved_carriers(self):
        automation = os.environ.get('VIDEO_BROWSER_AUTOMATION_MODULE')
        if not automation:
            self.skipTest('actual Chromium automation module required; browser-carrier acceptance unrun')
        result = subprocess.run(['bun', 'scripts/test-handheld-browser-carrier-acceptance.ts',
            '--prepared-root', str(self.prepared), '--automation-module', automation],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
            reference = decoded_source_reference(str(self.source / case['asset']['expected_file']), json.dumps(case))
            for frame, (ordinal, pts, ppm) in zip(receipt['frames'], reference):
                self.assertEqual(frame['selected_source_ordinal'], ordinal, 'receipt selected a non-nearest original frame')
                self.assertEqual(Decimal(str(frame['selected_source_pts'])), pts, 'receipt PTS is not decoded source PTS')
                self.assertEqual(p6_pixels((base / frame['path']).read_bytes()), p6_pixels(ppm), 'matching receipt hash does not prove original source pixels')

    def test_self_consistent_wrong_frame_or_pts_receipt_is_rejected(self):
        for case in self.expected:
            for mutation in ['pixels-and-hash', 'ordinal-and-pts', 'pixels-hash-ordinal-pts']:
                with self.subTest(case=case['id'], mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    base = output / case['id']
                    shutil.copytree(self.prepared / case['id'], base)
                    receipt_path = base / 'sampling.json'
                    receipt = json.loads(receipt_path.read_text())
                    first, other = receipt['frames'][:2]
                    self.assertNotEqual(first['sha256'], other['sha256'], 'actual source negative requires distinct frames')
                    if mutation != 'ordinal-and-pts':
                        replacement = (base / other['path']).read_bytes()
                        (base / first['path']).write_bytes(replacement)
                        first['sha256'] = hashlib.sha256(replacement).hexdigest()
                    if mutation != 'pixels-and-hash':
                        first['selected_source_ordinal'] = other['selected_source_ordinal']
                        first['selected_source_pts'] = other['selected_source_pts']
                    receipt_path.write_text(json.dumps(receipt))
                    self.assertTrue(self.verify(case, self.source, output), 'self-consistent wrong source correspondence was accepted')

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


class ApprovedAttributionAcceptance(unittest.TestCase):
    """Supplemental 4244038768 contract, authored against frozen 1c1dbdd."""
    def test_approved_source_attribution_and_integrity_are_canonical(self):
        expected = json.loads(ORACLE.read_text())
        self.assertEqual(checker.validate_manifest(expected), [])
        for original in expected['cases']:
            values = {
                'creator': 'Unrelated creator',
                'title': 'Unrelated source title',
                'source_url': 'https://example.invalid/unrelated-source',
                'download_url': 'https://example.invalid/unrelated-original.webm',
                'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                'owner_approval_url': 'https://github.com/moritzbrantner/video-to-3d/issues/74#unrelated',
                'sha256': '0' * 64,
                'byte_length': original['asset']['byte_length'] + 1,
                'expected_file': 'unrelated-original.webm',
                'mode': 'provider',
                'redistribute': False,
            }
            for field,value in values.items():
                with self.subTest(case=original['id'], field=field):
                    case = copy.deepcopy(original)
                    case['asset'][field] = value
                    document = copy.deepcopy(expected)
                    document['cases'] = [case]
                    errors = checker.validate_manifest(document)
                    self.assertTrue(errors, f'approved ID accepted different canonical asset.{field}')
                    reason = '; '.join(errors).lower()
                    aliases = {'sha256': ['source'], 'owner_approval_url': ['owner-approved'],
                               'license_url': ['cc by'], 'mode': ['manual']}
                    self.assertTrue(any(token in reason for token in [field] + aliases.get(field, [])),
                                    'unrelated generic rejection: ' + reason)

    def test_actual_public_preparation_rejects_false_attribution_before_output_use(self):
        source = os.environ.get('VIDEO_HANDHELD_ASSET_ROOT')
        if not source:
            self.skipTest('actual licensed originals required; public attribution boundary unrun')
        expected = json.loads(ORACLE.read_text())
        checks = [('bun', 0, 'creator', 'Unrelated creator'),
                  ('python', 1, 'download_url', 'https://example.invalid/unrelated-original.ogv')]
        for adapter,index,field,value in checks:
            with self.subTest(adapter=adapter,field=field), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                document = copy.deepcopy(expected)
                document['cases'] = [document['cases'][index]]
                case = document['cases'][0]
                case['asset'][field] = value
                original = Path(source) / expected['cases'][index]['asset']['expected_file']
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), expected['cases'][index]['asset']['sha256'])
                output = base / 'outputs'
                case_output = output / case['id']
                case_output.mkdir(parents=True)
                sentinel = case_output / 'retained.txt'
                sentinel.write_bytes(b'unchanged attribution acceptance evidence')
                before = [(str(path.relative_to(output)),path.read_bytes()) for path in output.rglob('*') if path.is_file()]
                modified = sentinel.stat().st_mtime_ns
                manifest = base / 'false-attribution.json'
                manifest.write_text(json.dumps(document))
                if adapter == 'bun':
                    command = ['bun','tools/run_real_video_case.ts','--case',case['id'],'--asset-root',source,
                               '--output-root',str(output),'--manifest',str(manifest),'--prepare-only']
                else:
                    command = ['python3','tools/check_real_video_manifest.py',str(manifest),
                               '--prepare-case',case['id'],'--asset-root',source,'--output-root',str(output)]
                result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0,'false attribution was prepared')
                reason = (result.stdout+result.stderr).lower()
                # The deliberately retained incomplete output bounds the old
                # baseline: a missing receipt is NOT attribution rejection.
                self.assertIn(field,reason,'public boundary admitted false attribution before unrelated failure: '+reason)
                self.assertTrue(any(token in reason for token in ['approved','canonical','pinned']),reason)
                self.assertEqual([(str(path.relative_to(output)),path.read_bytes()) for path in output.rglob('*') if path.is_file()],before)
                self.assertEqual(sentinel.stat().st_mtime_ns,modified)

    def test_actual_prepared_verification_rejects_false_source_metadata_first(self):
        source = os.environ.get('VIDEO_HANDHELD_ASSET_ROOT')
        prepared = os.environ.get('VIDEO_HANDHELD_PREPARED_ROOT')
        if not source or not prepared:
            self.skipTest('actual licensed originals/prepared receipts required; verifier boundary unrun')
        output = Path(prepared)
        before = sorted((str(path.relative_to(output)),path.stat().st_size,path.stat().st_mtime_ns)
                        for path in output.rglob('*') if path.is_file())
        self.assertTrue(before, 'actual prepared root must be populated')
        expected = json.loads(ORACLE.read_text())
        for original in expected['cases']:
            with self.subTest(case=original['id']):
                receipt = output / original['id'] / 'sampling.json'
                self.assertTrue(receipt.is_file(), 'actual prepared receipt missing')
                case = copy.deepcopy(original)
                case['asset']['source_url'] = 'https://example.invalid/unrelated-source'
                errors = checker.verify_prepared_case(case,Path(source),output)
                self.assertTrue(errors,'actual verified bytes were admitted with false source metadata')
                self.assertTrue(any('source_url' in error.lower() for error in errors),
                                'verifier did not reject canonical source metadata: '+str(errors))
                self.assertEqual(sorted((str(path.relative_to(output)),path.stat().st_size,path.stat().st_mtime_ns)
                                        for path in output.rglob('*') if path.is_file()),before)
