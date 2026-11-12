"""Scheduling isolation and no-overwrite checks; no GPU or learner imports."""
import json
from pathlib import Path
import tempfile
import unittest

from .config import configuration, run_id
from .priority_controller import choose_slot, initial_states


class PriorityTest(unittest.TestCase):
    def setUp(self):
        self.cpus = list(range(128))
        self.hardware = {g: dict(free_mib=30000, pids=[10 + g, 14 + g]) for g in range(4)}
        self.running = [dict(pid=10 + i, gpu=i % 4, cpus=list(range(i * 8, (i + 1) * 8)))
                        for i in range(8)]
        self.ibolt = configuration('ibolt', 'door', 5, 'e256-u025', temperature=.05)
        self.sac = configuration('sac', 'door', 5, 'e64-u1')

    def test_additional_slot_preserves_baseline_cpu_and_capacity(self):
        gpu, cpus = choose_slot(self.ibolt, self.running, self.hardware, self.cpus, 0)
        self.assertEqual((gpu, cpus), (0, list(range(64, 72))))
        self.assertIsNone(choose_slot(self.sac, self.running, self.hardware, self.cpus))

    def test_unknown_gpu_process_blocks_admission(self):
        self.hardware[0]['pids'].append(999)
        self.assertIsNone(choose_slot(self.ibolt, self.running, self.hardware, self.cpus, 0))

    def test_importing_process_counts_before_cuda_allocation(self):
        self.running.append(dict(pid=99, gpu=0, cpus=list(range(64, 72))))
        self.assertIsNone(choose_slot(self.ibolt, self.running, self.hardware, self.cpus, 0))

    def test_baseline_reuses_only_its_freed_slot(self):
        self.running = [s for s in self.running if s['pid'] != 10]
        self.hardware[0]['pids'].remove(10)
        self.running.append(dict(pid=99, gpu=0, cpus=list(range(64, 72))))
        self.hardware[0]['pids'].append(99)
        self.assertEqual(choose_slot(self.sac, self.running, self.hardware, self.cpus),
                         (0, list(range(8))))

    def test_migrated_job_never_overwrites_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); name = run_id(self.ibolt)
            (root / 'queue-state.json').write_text(json.dumps({'jobs': {}}))
            plan = dict(operation_commit='a' * 40, jobs=[dict(run_id=name,
                config=self.ibolt, origin='dilg-01', priority=0)])
            self.assertEqual(initial_states(root, plan)[name]['status'], 'pending')
            (root / 'runs' / name).mkdir(parents=True)
            with self.assertRaises(AssertionError):
                initial_states(root, plan)


if __name__ == '__main__':
    unittest.main()
