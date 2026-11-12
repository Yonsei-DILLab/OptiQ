"""Protect matched scientific settings, cohort identity and peer GPU slots."""
import json
from pathlib import Path
import tempfile
import unittest

from .collect_curves import collect
from .config import configuration, run_id
from .temp001_controller import campaign, slot_for


class Temperature001Test(unittest.TestCase):
    def test_only_temperature_changes_and_eight_distinct_runs(self):
        jobs = campaign(0) + campaign(1)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({run_id(c) for c in jobs}), 8)
        for c in jobs:
            ref = configuration('ibolt', c['task'], c['seed'], 'e256-u025', temperature=.10)
            self.assertEqual({k: v for k, v in c.items() if k != 'temperature'},
                             {k: v for k, v in ref.items() if k != 'temperature'})
            self.assertTrue(run_id(c).endswith('-t001'))
            self.assertNotEqual(run_id(c), run_id(ref))

    def test_recovery_slot_and_excluded_device_are_preserved(self):
        cpus = list(range(256))
        hardware = {g: dict(free_mib=30000, pids=[10 + g, 14 + g]) for g in range(4)}
        running = [dict(pid=10 + i, gpu=i % 4, cpus=list(range(i * 8, (i + 1) * 8)))
                   for i in range(8)]
        running.append(dict(pid=99, gpu=0, cpus=list(range(64, 72))))
        hardware[0]['pids'].append(99)
        c = campaign(0)[0]
        self.assertEqual(slot_for(c, running, hardware, cpus, 0), (2, list(range(80, 88))))
        hardware[2]['pids'].append(999)  # Unknown work must block even a free CPU slot.
        self.assertEqual(slot_for(c, running, hardware, cpus, 0), (3, list(range(88, 96))))

    def test_collector_includes_queued_temperature_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'priority-queue-state.json').write_text(json.dumps({'jobs': {'old': {'status': 'running'}}}))
            (root / 'temp001-queue-state.json').write_text(json.dumps({'jobs': {'new': {'status': 'pending'}}}))
            self.assertEqual(set(collect(root)['queue']['jobs']), {'old', 'new'})


if __name__ == '__main__':
    unittest.main()
