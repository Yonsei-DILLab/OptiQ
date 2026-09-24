"""Guard the approved positive sweep and the real model's regulator settings."""
import tempfile
from types import SimpleNamespace
import unittest

from .dacer_target import verify_target
from .register_dacer_positive import campaign_manifest, TARGETS, TASKS, TOTAL_STEPS
from .register_euclidean_no_step import campaign_manifest as control_manifest
from .settings import expected_updates


class PositiveDacerTests(unittest.TestCase):
    def test_complete_balanced_grid_preserves_other_learning_settings(self):
        controls={j['task']:j for part in (0,1)
                  for j in control_manifest('/unused','test',part)['jobs']}
        jobs=[]
        for shard in (0,1):
            manifest=campaign_manifest('/unused','test',shard)
            self.assertTrue(manifest['dacer_enabled'])
            self.assertEqual(manifest['wandb_project'],'antmaze')
            self.assertEqual(manifest['expected_regulator_updates'],32)
            self.assertEqual(len(manifest['jobs']),8)
            self.assertEqual({j['task'] for j in manifest['jobs'][:4]},set(TASKS))
            self.assertEqual(manifest['excluded_hosts'],['vast1'])
            self.assertFalse(manifest['automatic_restart'])
            for target in TARGETS:
                self.assertEqual(sum(j['dacer_target_entropy_per_dim']==target for j in manifest['jobs']),2)
            jobs.extend(manifest['jobs'])
        self.assertEqual(len({j['id'] for j in jobs}),16)
        self.assertEqual({(j['task'],j['dacer_target_entropy_per_dim']) for j in jobs},
                         {(task,target) for task in TASKS for target in TARGETS})
        self.assertEqual((TOTAL_STEPS,expected_updates(TOTAL_STEPS)),(508416,15632))
        for job in jobs:
            self.assertEqual(job['dacer'],'on')
            self.assertEqual(job['dacer_interval_updates'],500)
            self.assertEqual(job['steps'],TOTAL_STEPS)
            self.assertEqual(job['eval_interval'],100000)
            comparable=dict(job)
            for key in ('id','dacer','steps'):
                comparable[key]=controls[job['task']][key]
            for key in ('seed','eval_interval','dacer_target_entropy_per_dim','dacer_interval_updates'):
                comparable.pop(key)
            self.assertEqual(comparable,controls[job['task']])

    def test_live_interval_sign_and_non_ablation_settings(self):
        for target in TARGETS:
            cfg=SimpleNamespace(target_entropy_per_dim=target,behavior_only=True,
                initial_alpha=.27,alpha_lr=.03,interval_updates=500,components=3,
                samples=200,noise_scale=.1,entropy_seed=42)
            learner=SimpleNamespace(model=SimpleNamespace(regulator_cfg=cfg,regulator_enabled=True,
                action_space=SimpleNamespace(shape=(8,)),regulator_noise_std=.027),
                config={'dacer':{'target_entropy_per_dim':target}})
            with tempfile.TemporaryDirectory() as folder:
                proof=verify_target(learner,folder,target,500)
                self.assertAlmostEqual(proof['target_entropy'],8*target)
                with self.assertRaises(AssertionError):verify_target(learner,folder,target)
                with self.assertRaises(AssertionError):verify_target(learner,folder,-target,500)
                cfg.noise_scale=.2
                with self.assertRaises(AssertionError):verify_target(learner,folder,target,500)


if __name__=='__main__':unittest.main()
