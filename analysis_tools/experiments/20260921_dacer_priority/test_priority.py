import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import controller as c

class QueueTests(unittest.TestCase):
    def test_priority_without_completion_barrier(self):
        jobs=c.old.job_plan('ant','dacer',(.25,),(1.,))+c.old.job_plan('ant','beta',(.1,),(.5,.9))
        for j in jobs: j.update(status='queued',priority=int(j['stage']=='beta'))
        for j in jobs[:4]: j['status']='running'
        self.assertEqual(c.ordered_pending(jobs)[0]['seed'],4)
        jobs[4]['status']='running'
        self.assertEqual(c.ordered_pending(jobs)[0]['stage'],'beta')

    def test_selection_uses_matched_completed_seeds(self):
        jobs=[]
        for t,seed,value in ((.1,0,100),(.05,0,90),(.05,1,9999)):
            jobs.append(dict(temperature=t,seed=seed,wandb_url='test',metrics={
                'stochastic_z_last_100k_mean':value,'zero_z_last_100k_mean':0}))
        result=c.select_completed(jobs,'ant')
        self.assertEqual(result['matched_seeds'],[0])
        self.assertEqual(result['selected_temperature'],.1)

    def test_missing_final_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for mode in ('zero_z','stochastic_z'):
                np.savez(p/f'evaluations_{mode}.npz',timesteps=np.arange(905000,1000001,5000),results=np.ones((20,10)))
            self.assertEqual(c.old.summarize_run(p)['stochastic_z_last_100k_mean'],1)
            np.savez(p/'evaluations_stochastic_z.npz',timesteps=np.arange(905000,1000000,5000),results=np.ones((19,10)))
            with self.assertRaises(AssertionError): c.old.summarize_run(p)

    def test_dacer_default_parity(self):
        import train_dacer
        for task in ('ant','humanoid'):
            cfg=train_dacer.compose_config(task,4,'/tmp/config-only')
            self.assertTrue(cfg.dacer.enabled)
            self.assertEqual(cfg.dacer.target_entropy_per_dim,-.9)
            self.assertEqual(cfg.alg.actor.temperature,.25)
            self.assertEqual(cfg.alg.batch_size,256)

if __name__=='__main__': unittest.main()
