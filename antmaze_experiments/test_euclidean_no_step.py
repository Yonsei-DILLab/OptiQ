import copy
import unittest
import numpy as np
from .progress_reward import (EUCLIDEAN_NO_COST_PROFILE as PROFILE,
                              progress_reward, specification, maze_geometry)
from .register_euclidean_no_step import campaign_manifest
from .register_geodesic_no_step import campaign_manifest as geodesic_control
from .settings import REWARD_PROFILES, reward_description


class EuclideanNoStepTests(unittest.TestCase):
    def test_runtime_logging_accepts_every_reward_profile(self):
        for profile in REWARD_PROFILES:
            self.assertTrue(reward_description(profile))
        self.assertEqual(reward_description(PROFILE),
            'nearest-goal Euclidean distance decrease *100; no step penalty; no success bonus')

    def test_exact_euclidean_formula_and_cost_removal(self):
        for task in ('v1', 'v2', 'v3', 'v4'):
            goals = maze_geometry(task)[1]
            before = np.array([[0., 0.], [.1, .1], [-.2, .3]])
            after = np.array([[0., 0.], [-.2, .3], [.1, .1]])
            direct = lambda p: np.linalg.norm(p[:, None, :] - goals, axis=-1).min(-1)
            reward, _, _ = progress_reward(before, after, task, PROFILE, 20.)
            np.testing.assert_allclose(reward, 100 * (direct(before) - direct(after)))
            self.assertEqual(reward[0], 0.)
            self.assertAlmostEqual(reward[1] + reward[2], 0.)
            previous, _, _ = progress_reward(before, after, task,
                                             'progress100_euclidean_no_bonus', 20.)
            np.testing.assert_allclose(reward - previous, 1.)
            spec = specification(task, PROFILE)
            self.assertEqual(spec['formula'], '100*(d(current)-d(next))')
            self.assertEqual(spec['distance'], 'nearest-goal Euclidean')
            self.assertEqual(spec['step_cost'], 0.)
            self.assertEqual(spec['goal_bonuses'], [0] * len(goals))
            self.assertFalse(spec['success_bonus_enabled'])
            self.assertTrue(spec['success_terminates'])

    def test_terminal_bonus_ignored_and_distance_not_zeroed(self):
        for task in ('v1', 'v2', 'v3', 'v4'):
            for goal in maze_geometry(task)[1]:
                for bonus in (0., 10., 20.):
                    reward, _, remaining = progress_reward(goal + [.6, 0],
                        goal + [.3, 0], task, PROFILE, bonus)
                    self.assertAlmostEqual(float(reward), 30.)
                    self.assertAlmostEqual(float(remaining), .3)

    def test_equal_mirrored_progress_at_v3_v4_start(self):
        for task, other in (('v3', [-.5, .5]), ('v4', [.5, -.5])):
            right, _, _ = progress_reward([0, 0], [.5, .5], task, PROFILE, 0)
            # v3 goals are exchanged by point inversion; v4 by y reflection.
            if task == 'v3':
                right, _, _ = progress_reward([0, 0], [.5, -.5], task, PROFILE, 0)
            mirrored, _, _ = progress_reward([0, 0], other, task, PROFILE, 0)
            self.assertAlmostEqual(float(right), float(mirrored))

    def test_four_jobs_preserve_control_settings(self):
        jobs = []
        for shard in (0, 1):
            manifest = campaign_manifest('/unused', 'test', shard)
            control = geodesic_control('/unused', 'test', shard)
            self.assertEqual(manifest['wandb_project'], 'antmaze')
            self.assertEqual(manifest['backfill_seconds'], 2)
            for new, old in zip(manifest['jobs'], control['jobs']):
                candidate = copy.deepcopy(new)
                for field in ('id', 'reward_profile', 'reward_specification'):
                    candidate[field] = old[field]
                self.assertEqual(candidate, old)
                self.assertEqual(new['eval_starts'], 'upstream')
                self.assertEqual(new['reward_profile'], PROFILE)
                jobs.append(new)
        self.assertEqual([j['task'] for j in jobs], ['v3', 'v4', 'v1', 'v2'])


if __name__ == '__main__':
    unittest.main()
