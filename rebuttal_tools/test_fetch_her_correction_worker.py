import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from rebuttal_tools.fetch_her_correction_worker import corrected_worker_source, install_corrected_worker


ROOT = Path(__file__).resolve().parents[1]


class CorrectionAdapterTests(unittest.TestCase):
    def test_install_preserves_frozen_worker_bytes_and_does_not_start_training(self):
        worker = ROOT / 'rebuttal_tools/fetch_suite_5m_worker.py'
        before = worker.read_bytes()
        names = ['rebuttal_tools/fetch_suite_5m_worker.py',
                 'rebuttal_tools/fetch_her_correction_worker.py',
                 'rebuttal_tools/fetch_her_normalization.py']
        manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
        alias = 'rebuttal_tools.fetch_suite_5m_worker'
        previous = sys.modules.pop(alias, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'manifest.json'
                path.write_text(json.dumps(manifest))
                module = install_corrected_worker(dict(method='ibolt', source=str(ROOT), source_manifest=str(path)))
                self.assertTrue(callable(module.run))
                self.assertEqual(worker.read_bytes(), before)
                self.assertNotIn('model', module.__dict__)
        finally:
            sys.modules.pop(alias, None)
            if previous is not None:
                sys.modules[alias] = previous

    def test_incompatible_or_already_corrected_source_is_rejected(self):
        source = (ROOT / 'rebuttal_tools/fetch_suite_5m_worker.py').read_text()
        corrected = corrected_worker_source(source)
        with self.assertRaises(ValueError):
            corrected_worker_source(corrected)
        with self.assertRaises(ValueError):
            corrected_worker_source(source.replace('vec = VecNormalize(', 'vec = OtherNormalize('))

    def test_launch_from_a_different_source_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, 'frozen source snapshot'):
                install_corrected_worker(dict(method='sac', source=directory))


if __name__ == '__main__':
    unittest.main()
