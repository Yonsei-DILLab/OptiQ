"""Check the authorized grid and equality to the existing T1 run profile."""
import copy
import unittest

from .register_dacer_off import campaign_manifest as control_manifest
from .register_dacer_off_temperature import campaign_manifest, TASKS, TEMPERATURES
from .settings import BUDGETS, total_budget


class DacerOffTemperatureTests(unittest.TestCase):
    def test_only_temperature_changes_in_all_nine_jobs(self):
        controls = {j['task']: j for shard in (0, 1)
                    for j in control_manifest('/unused', 'test', shard)['jobs']}
        all_jobs = []
        for shard in (0, 1):
            manifest = campaign_manifest('/unused', 'test', shard)
            self.assertEqual(manifest['wandb_project'], 'antmaze')
            self.assertEqual(manifest['replay_capacity'], 1000000)
            self.assertFalse(manifest['dacer_enabled'])
            self.assertIsNone(manifest['gradient_clipping'])
            self.assertIsNone(manifest['policy_temperature_schedule'])
            self.assertEqual(manifest['backfill_seconds'], 2)
            self.assertTrue(manifest['failure_holds_pending'])
            self.assertFalse(manifest['automatic_restart'])
            self.assertEqual(sum(BUDGETS[j['task']] for j in manifest['jobs']), 18000000)
            for entry in manifest['jobs']:
                self.assertEqual(entry['steps'], total_budget(entry['task']))
                self.assertEqual(entry['dacer'], 'off')
                self.assertEqual(entry['reward_profile'], 'dense')
                self.assertEqual(entry['noveld'], 'off')
                comparable = copy.deepcopy(entry)
                comparable['id'] = controls[entry['task']]['id']
                comparable['temperature'] = 1.
                self.assertEqual(comparable, controls[entry['task']])
                all_jobs.append(entry)
        self.assertEqual(len(all_jobs), 9)
        self.assertEqual(len({j['id'] for j in all_jobs}), 9)
        self.assertEqual({(j['task'], j['temperature']) for j in all_jobs},
                         {(task, temperature) for task in TASKS for temperature in TEMPERATURES})


if __name__ == '__main__':
    unittest.main()
