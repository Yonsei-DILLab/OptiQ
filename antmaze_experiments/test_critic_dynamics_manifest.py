"""Campaign contract tests: catch missing conditions or accidental budget/default drift."""
import contextlib
import copy
import io
import itertools
import sys
import unittest
import numpy as np
from unittest.mock import patch

from .dynamics_profiles import get_profile
from .progress_reward import specification, progress_reward, maze_geometry
from .register_critic_dynamics import (
    CAMPAIGN, CONDITIONS, EVAL_INTERVAL, HOSTS, POST_WARMUP_BUDGET,
    REFERENCE_SOURCE, SHARDS, TOTAL_STEPS, campaign_manifest, main,
)
from .register_euclidean_no_step import campaign_manifest as reference_manifest
from .settings import NUM_ENVS, PREFLIGHT_STEPS, WARMUP, expected_updates


class CriticDynamicsManifestTests(unittest.TestCase):
    def manifests(self):
        return [campaign_manifest('/unused', 'committed-test-source', shard) for shard in (0, 1)]

    def test_exact_cartesian_experiment_and_balanced_independent_shards(self):
        manifests = self.manifests()
        jobs = [j for m in manifests for j in m['jobs']]
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({j['id'] for j in jobs}), 12)
        self.assertEqual({(j['task'], j['dynamics_profile']) for j in jobs},
                         set(itertools.product(('v3', 'v4'), CONDITIONS)))
        for shard, manifest in enumerate(manifests):
            self.assertEqual(manifest['host'], HOSTS[shard])
            self.assertEqual(manifest['jobs'][0]['dynamics_profile'], 'control')
            self.assertEqual([j['task'] for j in manifest['jobs']].count('v3'), 3)
            self.assertEqual([j['task'] for j in manifest['jobs']].count('v4'), 3)
            self.assertEqual({j['dynamics_profile'] for j in manifest['jobs']}, set(CONDITIONS))
            self.assertEqual(manifest['backfill_seconds'], 2)
            self.assertFalse(manifest['cross_job_completion_barriers'])
            self.assertFalse(manifest['automatic_restart'])
            self.assertFalse(manifest['resume_cancelled_jobs'])
            self.assertTrue(manifest['existing_training_preserved'])
            self.assertEqual(manifest['excluded_hosts'], ['vast1'])
        for condition in CONDITIONS:
            self.assertNotEqual(next(j['task'] for j in manifests[0]['jobs'] if j['dynamics_profile'] == condition),
                                next(j['task'] for j in manifests[1]['jobs'] if j['dynamics_profile'] == condition))

    def test_strict_post_warmup_budget_and_actor_delay_accounting(self):
        self.assertEqual(TOTAL_STEPS, 508416)
        self.assertEqual(TOTAL_STEPS-WARMUP, 500224)
        self.assertGreater(TOTAL_STEPS-WARMUP, POST_WARMUP_BUDGET)
        self.assertLessEqual(TOTAL_STEPS-WARMUP-NUM_ENVS, POST_WARMUP_BUDGET)
        self.assertEqual(expected_updates(TOTAL_STEPS), 15632)
        self.assertEqual(PREFLIGHT_STEPS, 8448)
        self.assertEqual(expected_updates(PREFLIGHT_STEPS), 8)
        checkpoints = [(k*EVAL_INTERVAL+NUM_ENVS-1)//NUM_ENVS*NUM_ENVS for k in range(1, 6)]
        self.assertEqual(checkpoints, [100096, 200192, 300032, 400128, 500224])
        self.assertLess(checkpoints[-1], TOTAL_STEPS)
        for manifest in self.manifests():
            for job in manifest['jobs']:
                self.assertEqual(job['steps'], TOTAL_STEPS)
                self.assertEqual(job['expected_critic_updates'], 15632)
                self.assertEqual(job['expected_actor_updates'],
                                 7816 if job['dynamics_profile'] == 'actordelay2' else 15632)

    def test_exact_condition_knobs_and_reward_metadata(self):
        expected = {
            'control': (1., 1., .005, 5e-4, 1),
            'scale02': (.2, .2, .005, 5e-4, 1),
            'ema01': (1., 1., .01, 5e-4, 1),
            'criticlr2': (1., 1., .005, 1e-3, 1),
            'actordelay2': (1., 1., .005, 5e-4, 2),
            'scale02ema01': (.2, .2, .01, 5e-4, 1),
        }
        fields = ('reward_multiplier', 'temperature', 'tau', 'critic_lr', 'policy_delay')
        for manifest in self.manifests():
            for job in manifest['jobs']:
                profile = get_profile(job['dynamics_profile'])
                self.assertEqual(tuple(profile[k] for k in fields), expected[job['dynamics_profile']])
                self.assertEqual(job['dynamics_settings'], profile)
                self.assertEqual(job['temperature'], profile['temperature'])
                self.assertEqual(job['optiq_profile']['critic_lr'], profile['critic_lr'])
                self.assertEqual(job['optiq_profile']['tau'], profile['tau'])
                self.assertEqual(job['optiq_profile']['policy_delay'], profile['policy_delay'])
                spec = specification(job['task'], profile['reward_profile'])
                self.assertEqual(job['reward_specification'], spec)
                self.assertEqual(spec['progress_scale'], 100*profile['reward_multiplier'])
                self.assertEqual(spec['step_cost'], 0.)
                self.assertFalse(spec['success_bonus_enabled'])
                self.assertTrue(spec['success_terminates'])

    def test_scaled_reward_is_pointwise_fifth_including_goal_and_retreat(self):
        rng = np.random.default_rng(39)
        for task in ('v3', 'v4'):
            before = rng.uniform(-16, 16, (100, 2))
            after = before + rng.normal(0, .1, (100, 2))
            for goal in maze_geometry(task)[1]:
                before = np.r_[before, [goal + [.6, 0], goal + [.3, 0]]]
                after = np.r_[after, [goal + [.3, 0], goal + [.6, 0]]]
            raw, d0, d1 = progress_reward(before, after, task,
                get_profile('control')['reward_profile'], 20.)
            scaled, s0, s1 = progress_reward(before, after, task,
                get_profile('scale02')['reward_profile'], 20.)
            np.testing.assert_allclose(scaled, raw * .2, atol=1e-12)
            np.testing.assert_array_equal(d0, s0)
            np.testing.assert_array_equal(d1, s1)

    def test_shared_control_defaults_and_evaluation_are_preserved(self):
        for manifest in self.manifests():
            self.assertEqual(manifest['campaign'], CAMPAIGN)
            self.assertEqual(manifest['comparison_source'], REFERENCE_SOURCE)
            self.assertEqual(manifest['source_commit'], 'committed-test-source')
            self.assertEqual(manifest['wandb_entity'], 'OptiQ')
            self.assertEqual(manifest['wandb_project'], 'antmaze')
            self.assertEqual((manifest['num_envs'], manifest['batch_size'], manifest['updates_per_vector_step']), (256, 4096, 8))
            self.assertEqual(manifest['replay_capacity'], 1000000)
            self.assertEqual((manifest['log_std_min'], manifest['log_std_max'], manifest['initial_log_std']), (-5., -1., -1.))
            self.assertFalse(manifest['dacer_enabled'])
            self.assertFalse(manifest['noveld_enabled'])
            for job in manifest['jobs']:
                self.assertEqual(job['seed'], 0)
                self.assertEqual(job['method'], 'optiq')
                self.assertEqual((job['dacer'], job['noveld']), ('off', 'off'))
                self.assertEqual(job['eval_starts'], 'upstream')
                self.assertEqual(job['eval_interval'], 100000)
                self.assertEqual((job['interim_eval_episodes'], job['final_eval_episodes']), (40, 100))
                self.assertTrue(job['save_intermediate_policy'])
                self.assertEqual(job['optiq_profile']['actor_lr'], 3e-4)
                self.assertEqual(job['optiq_profile']['actor_hidden_dims'], [256, 256, 256])
                self.assertEqual(job['optiq_profile']['critic_hidden_dims'], [256, 256, 256])

    def test_building_manifest_cannot_mutate_reference_or_other_profiles(self):
        before = reference_manifest('/unused', 'reference', 0)
        self.manifests()
        self.assertEqual(before, reference_manifest('/unused', 'reference', 0))
        baseline = copy.deepcopy(get_profile('control'))
        manifest = campaign_manifest('/unused', 'new', 0)
        manifest['jobs'][0]['dynamics_settings']['tau'] = 100
        self.assertEqual(get_profile('control'), baseline)

    def test_reserved_host_and_mismatched_host_fail_before_registration(self):
        for argv in (['register', '--shard', '0', '--host', 'vast1'],
                     ['register', '--shard', '0', '--host', 'vast-heechan-199']):
            with patch.object(sys, 'argv', argv), patch('subprocess.check_output') as read_git:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    main()
                self.assertEqual(stopped.exception.code, 2)
                read_git.assert_not_called()


if __name__ == '__main__':
    unittest.main()
