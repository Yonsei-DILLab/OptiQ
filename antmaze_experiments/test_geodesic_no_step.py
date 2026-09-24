import copy
import unittest
from .progress_reward import NO_COST_PROFILE, progress_reward, specification, distance
from .register_geodesic_no_step import campaign_manifest
from .register_dacer_off import campaign_manifest as control


class NoStepTests(unittest.TestCase):
    def test_difference_is_exactly_one_for_all_mazes(self):
        for task in ('v1','v2','v3','v4'):
            for before,after in (([0,0],[0,0]),([0,0],[.1,.1]),([.1,.1],[0,0])):
                reward,*_=progress_reward(before,after,task,NO_COST_PROFILE,20.)
                original,*_=progress_reward(before,after,task,'progress100_geodesic_no_bonus',20.)
                self.assertAlmostEqual(float(reward-original),1.)
                self.assertAlmostEqual(float(reward),100.*(distance(before,task,NO_COST_PROFILE)-distance(after,task,NO_COST_PROFILE)))
            spec=specification(task,NO_COST_PROFILE)
            self.assertEqual(spec['step_cost'],0)
            self.assertEqual(spec['progress_scale'],100)
            self.assertFalse(spec['success_bonus_enabled'])
            self.assertTrue(spec['success_terminates'])

    def test_terminal_bonus_is_ignored(self):
        for bonus in (0,10,20):
            reward,_,terminal_distance=progress_reward([-7.4,0],[-7.7,0],'v1',NO_COST_PROFILE,bonus)
            self.assertAlmostEqual(float(reward),30.)
            self.assertAlmostEqual(float(terminal_distance),.3)

    def test_four_jobs_only_reward_changes(self):
        controls={j['task']:j for part in (0,1) for j in control('/unused','test',part)['jobs']}
        jobs=[]
        for shard in (0,1):
            m=campaign_manifest('/unused','test',shard)
            self.assertEqual(m['wandb_project'],'antmaze')
            self.assertEqual(m['backfill_seconds'],2)
            for j in m['jobs']:
                k=copy.deepcopy(j);k.pop('reward_specification')
                k['id']=controls[j['task']]['id'];k['reward_profile']='dense'
                self.assertEqual(k,controls[j['task']])
                jobs.append(j)
        self.assertEqual([j['task'] for j in jobs],['v3','v4','v1','v2'])


if __name__=='__main__': unittest.main()
