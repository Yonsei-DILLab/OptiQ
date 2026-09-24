"""Verify complete reward grid and equality to the T1 learning control."""
import copy
import unittest
from .register_dacer_off import campaign_manifest as controls
from .register_progress_factorial import campaign_manifest,TASKS,PROFILES
from .settings import total_budget

class FactorialTests(unittest.TestCase):
    def test_complete_grid_only_reward_changes(self):
        control={j['task']:j for part in (0,1) for j in controls('/unused','test',part)['jobs']}
        jobs=[]
        for part in (0,1):
            m=campaign_manifest('/unused','test',part)
            self.assertNotIn('reward_profile',m)
            self.assertEqual(m['wandb_project'],'antmaze')
            self.assertEqual(m['replay_capacity'],1000000)
            self.assertEqual(m['backfill_seconds'],2)
            self.assertFalse(m['automatic_restart'])
            self.assertTrue(m['failure_holds_pending'])
            for j in m['jobs']:
                self.assertEqual(j['steps'],total_budget(j['task']))
                comparable=copy.deepcopy(j)
                comparable.pop('reward_specification')
                comparable['id']=control[j['task']]['id']
                comparable['reward_profile']='dense'
                self.assertEqual(comparable,control[j['task']])
                spec=j['reward_specification']
                self.assertEqual(spec['success_bonus_enabled'],not j['reward_profile'].endswith('_no_bonus'))
                self.assertTrue(spec['success_terminates'])
                jobs.append(j)
        self.assertEqual(len(jobs),16)
        self.assertEqual(len({j['id'] for j in jobs}),16)
        self.assertEqual({(j['task'],j['reward_profile']) for j in jobs},
                         {(t,p) for t in TASKS for p in PROFILES})

if __name__=='__main__':unittest.main()
