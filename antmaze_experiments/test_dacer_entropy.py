"""Guard signed, per-dimension targets and preservation of the T=1 control."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from .register_dacer_entropy import campaign_manifest,TARGETS,TASKS
from .register_dense_t1 import campaign_manifest as fixed_t1
from .dacer_target import verify_target


class DacerEntropyTests(unittest.TestCase):
    def test_complete_sweep_without_duplicate_jobs_or_other_learning_changes(self):
        jobs=[]
        controls={entry['task']:entry for s in (0,1) for entry in fixed_t1('/unused','test',s)['jobs']}
        for shard,count in [(0,8),(1,7)]:
            manifest=campaign_manifest('/unused','test',shard)
            self.assertEqual(len(manifest['jobs']),count)
            jobs.extend(manifest['jobs'])
            self.assertEqual(manifest['wandb_project'],'antmaze')
        self.assertEqual(len({j['id'] for j in jobs}),15)
        self.assertEqual({(j['task'],j['dacer_target_entropy_per_dim']) for j in jobs},
                         {(t,h) for t in TASKS for h in TARGETS})
        for job in jobs:
            comparable=dict(job)
            comparable['id']=controls[job['task']]['id']
            comparable.pop('dacer_target_entropy_per_dim')
            self.assertEqual(comparable,controls[job['task']])
            self.assertNotIn('temperature_schedule',job)

    def test_live_signed_target_and_action_dimension(self):
        for target in TARGETS:
            c=SimpleNamespace(target_entropy_per_dim=target,behavior_only=True,
                initial_alpha=.27,alpha_lr=.03,interval_updates=10000,components=3,
                samples=200,noise_scale=.1,entropy_seed=42)
            learner=SimpleNamespace(model=SimpleNamespace(regulator_cfg=c,regulator_enabled=True,
                action_space=SimpleNamespace(shape=(8,)),regulator_noise_std=.027),
                config={'dacer':{'target_entropy_per_dim':target}})
            with tempfile.TemporaryDirectory() as folder:
                proof=verify_target(learner,folder,target)
                self.assertAlmostEqual(proof['target_entropy'],target*8)
                self.assertTrue((Path(folder)/'dacer-target-verification.json').exists())
                with self.assertRaises(AssertionError):verify_target(learner,folder,-target)


if __name__=='__main__':unittest.main()
