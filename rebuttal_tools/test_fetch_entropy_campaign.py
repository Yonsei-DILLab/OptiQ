"""Protect equal trial budgets, seed separation and preset selection rules."""
import tempfile
from pathlib import Path
import unittest
import json
from rebuttal_tools.fetch_entropy_campaign import GRIDS,prepare_jobs,score_pair,ranking,promote


class CampaignTests(unittest.TestCase):
    def test_equal_trials_and_disjoint_seeds(self):
        plan=dict(methods=list(GRIDS),source='/source',source_manifest='/manifest',
            torch_python='/torch',jax_python='/jax',ibolt_reference_config='/reference',folder='/output')
        prepare_jobs(plan)
        self.assertEqual(len(plan['jobs']),48)
        for method in GRIDS:
            for task in ('slide','reach'):
                rows=[j for j in plan['jobs'] if j['method']==method and j['task']==task]
                self.assertEqual(len(rows),6)
                self.assertEqual({j['seed'] for j in rows},{5,6})
                self.assertEqual(len({j['value'] for j in rows}),3)

    def test_score_requires_independent_complete_pairs(self):
        rows=[dict(seed=s,complete=True,steps=5000192,updates=623424,
            regular=[.1]*5+[.8]*5,final100=.75,final_episodes=100) for s in (5,6)]
        self.assertEqual(score_pair(rows)['late5'],.8)
        for key,value in [('seed',0),('updates',100),('final_episodes',10),('complete',False)]:
            changed=[dict(r) for r in rows];changed[0][key]=value
            with self.assertRaises(AssertionError):score_pair(changed)

    def test_identical_scores_prefer_existing_default(self):
        for method,(name,grid,default) in GRIDS.items():
            candidates=[dict(value=v,late5=1.,final100=1.,auc=1.,worst_late5=1.) for v in grid]
            self.assertEqual(max(candidates,key=lambda r:ranking(r,method))['value'],default)

    def test_promotion_creates_five_new_runs_once(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=dict(methods=['sac'],source='/source',source_manifest='/manifest',
                torch_python='/torch',jax_python='/jax',ibolt_reference_config='/reference',folder=directory,selections={})
            prepare_jobs(plan)
            pilot_paths=set()
            for j in plan['jobs']:
                if j['task']!='slide':continue
                j['status']='completed';p=Path(j['output']);p.mkdir(parents=True);pilot_paths.add(str(p))
                value=.9 if j['value']==-2 else .5
                records=[dict(steps=i*5000,primary_mode='policy',policy=dict(episodes=10,success_rate=value)) for i in range(1,7)]
                records.append(dict(steps=5000192,primary_mode='policy',policy=dict(episodes=100,success_rate=value)))
                (p/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                (p/'verified-completion.json').write_text(json.dumps(dict(completed=True,steps=5000192,updates=623424)))
            promote(plan);promote(plan)
            final=[j for j in plan['jobs'] if j['phase']=='final']
            self.assertEqual(len(final),5)
            self.assertEqual({j['seed'] for j in final},{0,1,2,3,4})
            self.assertEqual({j['value'] for j in final},{-2})
            self.assertFalse(pilot_paths.intersection(j['output'] for j in final))


if __name__=='__main__':unittest.main()
