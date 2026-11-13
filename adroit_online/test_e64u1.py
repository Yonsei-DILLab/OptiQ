"""Guard scientific equivalence, authorized scope and collision-free placement."""
import unittest

from .config import configuration, run_id
from .e64u1_controller import GPUS, campaign, slot_for


class E64U1Test(unittest.TestCase):
    def test_twelve_fresh_runs_change_only_env_and_utd(self):
        jobs = campaign()
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({run_id(c) for c in jobs}), 12)
        changes = {'profile', 'num_envs', 'utd', 'expected_updates'}
        for c in jobs:
            old = configuration('ibolt', c['task'], c['seed'], 'e256-u025', temperature=c['temperature'])
            self.assertEqual({k for k in c if c[k] != old[k]}, changes)
            self.assertEqual(c['gradient_steps'], 64)
            self.assertEqual(c['expected_updates'], 2989824)
            self.assertEqual((c['steps'] - c['warmup']) // 64 * c['gradient_steps'], c['expected_updates'])
            self.assertNotEqual(run_id(c), run_id(old))
            smoke = configuration('ibolt', c['task'], 4, 'e64-u1', smoke=True, temperature=c['temperature'])
            self.assertEqual(smoke['expected_updates'], 2048)
        for task, profile, temperature in [('pen', 'e64-u1', .10), ('relocate', 'e64-u1', .25),
                                          ('door', 'e64-u1', .01), ('hammer', 'e256-u1', .10)]:
            with self.assertRaises(AssertionError):
                configuration('ibolt', task, 4, profile, temperature=temperature)

    def test_gpu_and_cpu_admission(self):
        hardware = {g: dict(uuid=u, free_mib=32000, pids=[]) for g, u in GPUS.items()}
        cpus = list(range(64)); running = []
        for i in range(8):
            gpu, allocation = slot_for(running, hardware, cpus, list(GPUS))
            running.append(dict(pid=100 + i, gpu=gpu, cpus=allocation))
            hardware[gpu]['pids'].append(100 + i)
        self.assertIsNone(slot_for(running, hardware, cpus, list(GPUS)))
        self.assertEqual(len({c for s in running for c in s['cpus']}), 64)
        hardware[0]['pids'].append(999)
        self.assertIsNone(slot_for([], hardware, cpus, [0]))
        hardware[1]['pids'] = []; hardware[1]['free_mib'] = 12000
        self.assertIsNone(slot_for([], hardware, cpus, [1]))
        hardware[2]['pids'] = []
        self.assertEqual(slot_for([], hardware, cpus, [2, 3])[0], 2)
        self.assertIsNone(slot_for([], hardware, cpus, [2, 3], preferred=1))


if __name__ == '__main__':
    unittest.main()
