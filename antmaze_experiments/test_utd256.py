"""Verify the costly UTD=1 campaign before any GPU launch."""
import unittest

from .collection_profile import aligned_eval_step, expected_updates
from .progress_reward import EUCLIDEAN_NO_COST_PROFILE
from .register_utd256 import campaign_manifest
from .settings import WARMUP, total_budget


class Utd256CampaignTests(unittest.TestCase):
    def test_four_jobs_and_accounting(self):
        for condition, reward, dimensions, critic_lr in (
            ('control', EUCLIDEAN_NO_COST_PROFILE, [256, 256, 256], 5e-4),
            ('basic', 'sparse', [256, 256], 3e-4),
        ):
            jobs = [job for shard in (0, 1)
                    for job in campaign_manifest('/unused', 'sha', shard, condition)['jobs']]
            self.assertEqual({job['task'] for job in jobs}, {'v1', 'v2', 'v3', 'v4'})
            self.assertEqual(len({job['id'] for job in jobs}), 4)
            for job in jobs:
                self.assertEqual(job['collection_profile'], 'env256-update256')
                self.assertEqual(job['steps'], total_budget(job['task']))
                self.assertEqual(job['reward_profile'], reward)
                self.assertEqual(job['optiq_profile']['actor_hidden_dims'], dimensions)
                self.assertEqual(job['optiq_profile']['critic_hidden_dims'], dimensions)
                self.assertEqual(job['optiq_profile']['critic_lr'], critic_lr)
                self.assertEqual((job['temperature'], job['dacer'], job['noveld']),
                                 (1., 'off', 'off'))
                self.assertEqual((job['eval_interval'], job['interim_eval_episodes']),
                                 (50000, 40))
                self.assertTrue(job['save_intermediate_policy'])
                self.assertEqual(expected_updates(job['steps'], 'env256-update256'),
                                 job['steps'] - WARMUP)
                self.assertEqual(aligned_eval_step(1, job['eval_interval']), 50176)


if __name__ == '__main__':
    unittest.main()
