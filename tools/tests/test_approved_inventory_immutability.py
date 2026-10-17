"""Independent 4244183071 regression; fixed source review never follows custom manifests."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / 'tools/tests/fixtures/approved-handheld-expectations.json'
spec = importlib.util.spec_from_file_location('inventory_corpus_checker', ROOT / 'tools/check_real_video_manifest.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def mutations(case):
    """Every mutation preserves structural admission and changes only reviewed inventory."""
    original = case['preparation']['inventory']
    changes = []
    def change(label, action):
        inventory = copy.deepcopy(original)
        action(inventory)
        altered = copy.deepcopy(case)
        altered['preparation']['inventory'] = inventory
        changes.append((label, altered))
    change('object-id', lambda i: i['objects'][0].update(id='unreviewed-object'))
    change('object-visibility', lambda i: i['objects'][0].update(visible_samples=[next(n for n in range(18) if n not in i['objects'][0]['visible_samples'])]))
    change('object-coverage', lambda i: i['objects'][0].update(coverage='limited' if i['objects'][0]['coverage']=='repeated' else 'repeated'))
    change('object-uncertainty', lambda i: i['objects'][0].update(uncertainty='Unreviewed coverage assertion'))
    change('removed-object', lambda i: i['objects'].pop())
    change('added-object', lambda i: i['objects'].append({**copy.deepcopy(i['objects'][0]), 'id':'unreviewed-extra-object'}))
    change('architecture-id', lambda i: i['architecture'][0].update(id='unreviewed-architecture'))
    # All18 architecture visibility is legitimate for H3; narrower visibility is structurally valid.
    change('architecture-visibility', lambda i: i['architecture'][0].update(visible_samples=[17] if i['architecture'][0]['visible_samples'] != [17] else [0]))
    change('removed-architecture', lambda i: i['architecture'].pop())
    change('added-architecture', lambda i: i['architecture'].append({**copy.deepcopy(i['architecture'][0]), 'id':'unreviewed-extra-architecture'}))
    change('removed-dynamic-exclusion', lambda i: i['excluded_dynamic'].pop())
    change('added-dynamic-exclusion', lambda i: i['excluded_dynamic'].append('unreviewed-static-object'))
    return changes


class ApprovedInventoryImmutability(unittest.TestCase):
    def document(self, case):
        document = json.loads(ORACLE.read_text())
        document['cases'] = [case]
        return document

    def assert_reason(self, reason, case_id):
        reason = reason.lower()
        self.assertIn(case_id, reason)
        self.assertIn('inventory', reason)
        self.assertTrue(any(word in reason for word in ['approved','canonical','fixed','pinned']), reason)

    def test_exact_reviewed_inventory_remains_accepted_in_both_public_metadata_modes(self):
        for case in json.loads(ORACLE.read_text())['cases']:
            self.assertEqual(checker.validate_manifest(self.document(case)), [])
            self.assertEqual(checker.validate_selected_case_approval(self.document(case), case['id']), [])
            with tempfile.TemporaryDirectory() as directory:
                manifest = Path(directory)/'approved.json'
                manifest.write_text(json.dumps(self.document(case)))
                for extra in [[], ['--validate-case-approval',case['id']]]:
                    result = subprocess.run(['python3','tools/check_real_video_manifest.py',str(manifest),*extra],cwd=ROOT,capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_full_inventory_is_immutable_in_both_existing_public_validation_apis(self):
        admitted = []
        for case in json.loads(ORACLE.read_text())['cases']:
            for label, altered in mutations(case):
                self.assertNotEqual(altered['preparation']['inventory'],case['preparation']['inventory'])
                for mode, errors in [('normal',checker.validate_manifest(self.document(altered))),
                                     ('selected',checker.validate_selected_case_approval(self.document(altered),case['id']))]:
                    if not errors:
                        admitted.append((case['id'],label,mode))
                    else:
                        self.assert_reason('; '.join(errors),case['id'])
        self.assertEqual(admitted,[],f'custom manifests retuned fixed source inventory: {admitted}')

    def test_actual_cli_normal_and_selected_approval_reject_all_inventory_retuning(self):
        admitted = []
        for case in json.loads(ORACLE.read_text())['cases']:
            for label, altered in mutations(case):
                with tempfile.TemporaryDirectory() as directory:
                    manifest = Path(directory)/'retuned.json'
                    manifest.write_text(json.dumps(self.document(altered)))
                    for mode,extra in [('normal',[]),('selected',['--validate-case-approval',case['id']])]:
                        result = subprocess.run(['python3','tools/check_real_video_manifest.py',str(manifest),*extra],cwd=ROOT,capture_output=True,text=True,timeout=30)
                        if result.returncode==0:
                            admitted.append((case['id'],label,mode))
                        else:
                            self.assert_reason(result.stdout+result.stderr,case['id'])
        self.assertEqual(admitted,[],f'actual public metadata CLI accepted retuned inventory: {admitted}')

    def test_direct_prepare_and_verify_reject_inventory_before_unrelated_file_io(self):
        # Missing source/receipt roots isolate metadata eligibility: no fake prepared bytes.
        for case in json.loads(ORACLE.read_text())['cases']:
            altered = mutations(case)[0][1]
            for mode in ['prepare', 'verify']:
                with self.subTest(case=case['id'], mode=mode), tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    source, output = base/'absent-source', base/'absent-output'
                    if mode == 'prepare':
                        try:
                            checker.prepare_case(altered,source,output)
                        except (ValueError, OSError) as error:
                            reason = str(error)
                        else:
                            self.fail('retuned inventory was prepared')
                    else:
                        errors = checker.verify_prepared_case(altered,source,output)
                        self.assertTrue(errors)
                        reason = '; '.join(errors)
                    self.assertFalse(source.exists())
                    self.assertFalse(output.exists())
                    self.assert_reason(reason,case['id'])


if __name__ == '__main__':
    unittest.main()
