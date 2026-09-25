"""Check the reward-only UTD=1 queue before registration."""
import copy
import unittest

import numpy as np

from .progress_reward import (
    EUCLIDEAN_SCALE10_PROFILE,
    EUCLIDEAN_SCALE100_COST01_PROFILE,
    EUCLIDEAN_SCALE10_COST01_PROFILE,
    maze_geometry,
    progress_reward,
    specification,
)
from .register_utd256 import campaign_manifest as control_manifest
from .register_utd256_reward_grid import REWARDS, campaign_manifest
from .settings import reward_description


class Utd256RewardGridTests(unittest.TestCase):
    def test_exact_reward_formulas_and_unchanged_goal_termination(self):
        profiles = (
            (EUCLIDEAN_SCALE10_PROFILE, 10., 0., '10*(d(current)-d(next))'),
            (EUCLIDEAN_SCALE100_COST01_PROFILE, 100., .1,
             '100*(d(current)-d(next))-0.1'),
            (EUCLIDEAN_SCALE10_COST01_PROFILE, 10., .1,
             '10*(d(current)-d(next))-0.1'),
        )
        for task in ('v1', 'v2', 'v3', 'v4'):
            goals = maze_geometry(task)[1]
            before = np.array([[0., 0.], [.2, -.4]])
            after = np.array([[0., 0.], [.4, -.1]])
            d_before = np.linalg.norm(before[:, None] - goals, axis=-1).min(-1)
            d_after = np.linalg.norm(after[:, None] - goals, axis=-1).min(-1)
            for profile, scale, cost, formula in profiles:
                reward, _, _ = progress_reward(before, after, task, profile, 20.)
                np.testing.assert_allclose(reward, scale * (d_before-d_after) - cost)
                self.assertEqual(specification(task, profile)['formula'], formula)
                self.assertEqual(specification(task, profile)['step_cost'], cost)
                self.assertEqual(specification(task, profile)['goal_bonuses'], [0]*len(goals))
                self.assertTrue(specification(task, profile)['success_terminates'])
                self.assertTrue(reward_description(profile))
        self.assertEqual(reward_description('dense'),
                         'negative Euclidean distance from next xy to nearest goal; no sparse bonus')

    def test_manifest_changes_only_reward_from_live_control(self):
        seen = set()
        originals = {job['task']: job for shard in (0, 1)
                     for job in control_manifest('/unused', 'sha', shard, 'basic_euclidean')['jobs']}
        for shard in (0, 1, 2):
            manifest = campaign_manifest('/unused', 'sha', shard)
            self.assertEqual(len(manifest['jobs']), 4 if shard == 2 else 6)
            self.assertEqual(manifest['wandb_project'], 'antmaze')
            self.assertEqual(manifest['updates_per_transition'], 1.)
            self.assertEqual(bool(manifest['priority_campaign']), shard != 2)
            for job in manifest['jobs']:
                label, reward = next((label, reward) for label, reward in REWARDS
                                     if job['id'] == f"{job['task']}-optiq-utd1-{label}-s0")
                seen.add((job['task'], reward))
                expected = copy.deepcopy(originals[job['task']])
                expected['id'] = job['id']
                expected['reward_profile'] = reward
                expected.pop('reward_specification', None)
                if reward != 'dense':
                    expected['reward_specification'] = specification(job['task'], reward)
                self.assertEqual(job, expected)
        self.assertEqual(len(seen), 16)


if __name__ == '__main__':
    unittest.main()
