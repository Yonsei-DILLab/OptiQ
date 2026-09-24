"""Reward-only geometry, symmetry, and legacy compatibility checks."""
from pathlib import Path
import subprocess
import types
import unittest
import numpy as np
from . import progress_reward as reward
from .settings import reward_description


class StartNormalizedRewardTests(unittest.TestCase):
    def test_two_job_screen_changes_only_reward_and_explicit_temperature(self):
        from .register_start_normalized_geodesic import campaign_manifest
        from .register_geodesic_gamma_screen import campaign_manifest as old_manifest
        control = next(j for j in old_manifest('/unused', 'test')['jobs'] if j['task'] == 'v3')
        manifest = campaign_manifest('/unused', 'test')
        self.assertEqual(manifest['host'], 'vast-heechan-180')
        self.assertEqual(manifest['excluded_hosts'], ['vast1'])
        self.assertEqual(len(manifest['jobs']), 2)
        self.assertEqual(manifest['expected_learner_updates'], 7816)
        self.assertEqual(manifest['total_transitions_per_job'], 258304)
        self.assertEqual(manifest['wandb_project'], 'antmaze')
        self.assertFalse(manifest['automatic_restart'])
        for job in manifest['jobs']:
            changed = {k for k in set(job) | set(control) if job.get(k) != control.get(k)}
            expected = {'id', 'hypothesis', 'reward_profile', 'reward_specification'}
            if job['temperature'] == 3.:
                expected.add('temperature')
            else:
                self.assertEqual(job['temperature'], 1.)
            self.assertEqual(changed, expected)
            self.assertEqual(job['eval_starts'], 'upstream')
            self.assertEqual(job['reward_profile'], reward.START_NORMALIZED_PROFILE)

    def test_v3_known_reference_and_both_initial_directions(self):
        remaining, scale, weights = reward.goal_reference_scales('v3')
        np.testing.assert_allclose(remaining, [17.486460085480267, 16.47056274847832], atol=1e-10)
        self.assertAlmostEqual(scale, 16.47056274847832)
        self.assertTrue(np.all(weights <= 1))
        points = np.array([[0.,0.],[-.1,.1],[.1,-.1]])
        old, *_ = reward.progress_reward(points[0], points[1:], 'v3', reward.NO_COST_PROFILE, 0)
        new, *_ = reward.progress_reward(points[0], points[1:], 'v3', reward.START_NORMALIZED_PROFILE, 0)
        np.testing.assert_allclose(old, [-14.1421356237,14.1421356237], atol=1e-8)
        np.testing.assert_allclose(new, [12.9180051986,14.1421356237], atol=1e-8)
        self.assertTrue(np.all(new > 0))

    def test_same_goal_total_without_route_bonus_or_time_cost(self):
        p = reward.START_NORMALIZED_PROFILE
        for task in ('v1','v2','v3','v4'):
            _, scale, _ = reward.goal_reference_scales(task)
            goals = reward.maze_geometry(task)[1]
            endpoints = goals + np.array([.3,0.])
            np.testing.assert_allclose(reward.distance(endpoints, task, p), 0, atol=1e-10)
            returns, *_ = reward.progress_reward(np.zeros(2), endpoints, task, p, 12345)
            np.testing.assert_allclose(returns, np.full(len(goals), 100*scale))
            stop, *_ = reward.progress_reward(endpoints, endpoints, task, p, 12345)
            np.testing.assert_array_equal(stop, np.zeros(len(goals)))
            self.assertFalse(reward.bonus_enabled(p)); self.assertEqual(reward.step_cost(p), 0.)
            meta = reward.specification(task, p)
            self.assertFalse(meta['episode_start_dependent'])
            self.assertEqual(meta['success_radius'], .5)
            self.assertTrue(meta['success_terminates'])

    def test_conservative_cycle_and_same_state_independent_of_history(self):
        p = reward.START_NORMALIZED_PROFILE
        path = np.array([[0.,0.],[-.1,.1],[-.2,.1],[0.,0.],[.1,-.1],[0.,0.]])
        values, *_ = reward.progress_reward(path[:-1], path[1:], 'v3', p, np.zeros(len(path)-1))
        self.assertAlmostEqual(float(values.sum()), 0., places=10)
        self.assertEqual(float(reward.distance(path[0], 'v3', p)), float(reward.distance(path[-1], 'v3', p)))
        self.assertIn('start-normalized', reward_description(p))

    def test_existing_profiles_are_numerically_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(['git','-C',str(root),'show',
            '8e9d7d3c2c79f797654ccfb21913d2c000b89f71:antmaze_experiments/progress_reward.py'])
        old = types.ModuleType('old_progress_reward')
        old.__file__ = str(Path(reward.__file__))
        exec(compile(source, old.__file__, 'exec'), old.__dict__)
        points = np.array([[0.,0.],[-.1,.1],[.1,-.1]])
        for task in ('v1','v2','v3','v4'):
            for profile in old.PROFILES:
                np.testing.assert_array_equal(reward.distance(points,task,profile), old.distance(points,task,profile))
                self.assertEqual(reward.specification(task,profile), old.specification(task,profile))
                self.assertEqual(reward.value_support(task,profile), old.value_support(task,profile))
                current = reward.progress_reward(points[:-1],points[1:],task,profile,[0.,10.])[0]
                previous = old.progress_reward(points[:-1],points[1:],task,profile,[0.,10.])[0]
                np.testing.assert_array_equal(current, previous)


if __name__ == '__main__':
    unittest.main()
