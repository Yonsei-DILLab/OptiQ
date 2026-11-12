"""A retry replaces one logical curve; failed samples are never spliced in."""
import json
from pathlib import Path
import tempfile
import unittest

from .collect_curves import collect


class RecoveryTest(unittest.TestCase):
    def test_retry_selection_keeps_original_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, retry = 'hammer', 'hammer-retry1'
            for name, step in [(original, 1000), (retry, 100)]:
                run = root / 'runs' / name
                run.mkdir(parents=True)
                (run / 'spec.json').write_text(json.dumps({'profile': 'e256-u025'}))
                (run / 'progress.json').write_text(json.dumps({'steps': step}))
            (root / 'priority-queue-state.json').write_text(json.dumps(
                {'jobs': {original: {'status': 'failed'}}}))
            (root / 'recovery-queue-state.json').write_text(json.dumps({'jobs': {
                original: {'status': 'running', 'attempt_id': retry,
                           'output': str(root / 'runs' / retry)}}}))
            result = collect(root)
            self.assertEqual(list(result['runs']), [original])
            self.assertEqual(result['runs'][original]['progress.json']['steps'], 100)
            self.assertEqual(result['queue']['jobs'][original]['status'], 'running')
            self.assertTrue((root / 'runs' / original / 'progress.json').exists())


if __name__ == '__main__':
    unittest.main()
