import json
from pathlib import Path
import tempfile
import unittest

from .campaign import make_plan, promote, score, ranking, COEFFICIENTS, TEMPERATURES


def history(values,coverage=1):
    return [dict(steps=step,official=dict(success=value,reachable_goals=coverage))
        for step,value in zip((200192,400128,600064,800000,1000192),values)]


class CampaignTests(unittest.TestCase):
    def test_cartesian_grid_and_disjoint_seeds(self):
        plan=make_plan('hard',Path('/output'),Path('/source'),'/python')
        self.assertEqual(len(plan['jobs']),18)
        actual={(j['coefficient'],j['temperature'],j['seed']) for j in plan['jobs']}
        self.assertEqual(actual,{(c,t,s) for c in COEFFICIENTS for t in TEMPERATURES for s in (5,6)})
        self.assertFalse(set(plan['selection_seeds'])&set(plan['final_seeds']))
        self.assertEqual(plan['fixed']['batch_size'],256)
        self.assertEqual(plan['fixed']['utd'],1)

    def test_stable_late_performance_beats_lucky_final_checkpoint(self):
        lucky=score([dict(seed=s,evaluations=history([0,0,0,0,1])) for s in (5,6)])
        stable=score([dict(seed=s,evaluations=history([0,0,.9,.9,.9])) for s in (5,6)])
        self.assertGreater(stable['late_success'],lucky['late_success'])
        with self.assertRaises(ValueError):
            score([dict(seed=0,evaluations=history([1]*5)),dict(seed=1,evaluations=history([1]*5))])

    def test_all_runs_required_and_selection_launches_exactly_five(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);plan=make_plan('medium',folder,folder/'source','/python')
            for job in plan['jobs']:
                out=Path(job['output']);out.mkdir(parents=True)
                values=[.95]*5 if job['coefficient']==-.25 and job['temperature']==.1 else [.4]*5
                (out/'evaluations.jsonl').write_text('\n'.join(json.dumps(r) for r in history(values,3))+'\n')
                job['status']='completed'
            plan['jobs'][0]['status']='running'
            promote(plan,folder)
            self.assertNotIn('selection',plan)
            plan['jobs'][0]['status']='completed'
            promote(plan,folder)
            selected=[j for j in plan['jobs'] if j['phase']=='selected-five-seed']
            self.assertEqual([j['seed'] for j in selected],[0,1,2,3,4])
            self.assertTrue(all(j['coefficient']==-.25 and j['temperature']==.1 for j in selected))
            promote(plan,folder)
            self.assertEqual(len(plan['jobs']),23)

    def test_no_observed_success_does_not_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);plan=make_plan('hard',folder,folder/'source','/python')
            for job in plan['jobs']:
                out=Path(job['output']);out.mkdir(parents=True)
                (out/'evaluations.jsonl').write_text('\n'.join(json.dumps(r) for r in history([0]*5,0))+'\n')
                job['status']='completed'
            promote(plan,folder)
            self.assertEqual(plan['selection']['status'],'needs_review_no_success')
            self.assertEqual(len(plan['jobs']),18)

    def test_failed_pair_is_disclosed_and_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);plan=make_plan('hard',folder,folder/'source','/python')
            for job in plan['jobs']:
                out=Path(job['output']);out.mkdir(parents=True)
                (out/'evaluations.jsonl').write_text('\n'.join(json.dumps(r) for r in history([.8]*5,3))+'\n')
                job['status']='completed'
            plan['jobs'][0].update(status='failed',error='synthetic numerical failure')
            promote(plan,folder)
            self.assertEqual(len(plan['selection']['excluded']),1)
            self.assertNotEqual((plan['selection']['best']['coefficient'],plan['selection']['best']['temperature']),(-1.,.01))
            self.assertEqual(len(plan['jobs']),23)


if __name__=='__main__':unittest.main()
