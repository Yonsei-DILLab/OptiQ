"""Scientific budget and cohort checks; simulator checks run in the GPU preflight."""
import unittest
from .config import campaign, configuration, run_id
from .controller import assigned
from .ibolt_controller import campaign as ibolt_campaign


class ProtocolTest(unittest.TestCase):
    def test_budget_and_shards(self):
        jobs = campaign()
        self.assertEqual(len(jobs), 48)
        self.assertEqual(len({run_id(c) for c in jobs}), 48)
        for c in jobs:
            self.assertEqual(sum(assigned(c, s) for s in (0, 1)), 1)
            blocks = (c['steps'] - c['warmup']) // c['num_envs']
            self.assertEqual(blocks * c['gradient_steps'], c['expected_updates'])
            self.assertIn(c['expected_updates'], (747456, 2989824))
            self.assertEqual(c['demonstration_ratio'], 0)
        for s in (0, 1):
            rows = [c for c in jobs if assigned(c, s) and c['profile'] == 'e256-u025']
            self.assertEqual(len(rows), 8)
            self.assertEqual({c['task'] for c in rows}, {'door', 'hammer', 'pen', 'relocate'})
            self.assertEqual({c['method'] for c in rows}, {'sac', 'td3'})

    def test_only_scientific_changes_are_environment_count_and_utd(self):
        baseline = configuration('sac', 'door', 4, 'e256-u025')
        variable = {'profile', 'num_envs', 'utd', 'gradient_steps', 'expected_updates'}
        for name in ('e256-u1', 'e64-u1'):
            other = configuration('sac', 'door', 4, name)
            self.assertEqual({k: v for k, v in baseline.items() if k not in variable},
                             {k: v for k, v in other.items() if k not in variable})

    def test_smoke_includes_complete_native_episodes(self):
        for p in ('e256-u025', 'e256-u1', 'e64-u1'):
            c = configuration('sac', 'pen', 4, p, smoke=True)
            self.assertGreaterEqual(c['steps'] // c['num_envs'], c['horizon'])
            self.assertEqual((c['steps'] - c['warmup']) * c['utd'], c['expected_updates'])

    def test_temperature_extension_preserves_matched_protocol_and_run_identity(self):
        original = ibolt_campaign()
        extra = ibolt_campaign((.05, .10))
        self.assertEqual(len(original), 8)
        self.assertEqual(len(extra), 16)
        self.assertEqual(len({run_id(c) for c in original + extra}), 24)
        for c in extra:
            reference = configuration('ibolt', c['task'], c['seed'], 'e256-u025')
            self.assertEqual({k: v for k, v in c.items() if k != 'temperature'},
                             {k: v for k, v in reference.items() if k != 'temperature'})
            self.assertEqual(c['expected_updates'], 747456)
        self.assertEqual(run_id(original[0]), 'door-ibolt-s4-e256-u025-t025')
        self.assertEqual({c['temperature'] for c in extra[:4]}, {.05, .10})
        self.assertEqual({c['task'] for c in extra[:4]}, {'door', 'hammer', 'pen', 'relocate'})
        with self.assertRaises(AssertionError):
            configuration('sac', 'door', 4, 'e256-u025', temperature=.05)


if __name__ == '__main__':
    unittest.main()
