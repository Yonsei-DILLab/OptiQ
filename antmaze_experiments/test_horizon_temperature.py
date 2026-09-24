"""Guard hypothesis isolation, budget and queue priority without a finish barrier."""
import json
from pathlib import Path
import tempfile
import unittest
from .register_horizon_temperature import campaign_manifest, CONDITIONS, TOTAL_STEPS
from .register_dacer_positive import campaign_manifest as reference
from .controller import priority_queue_ready
from .settings import expected_updates


class HorizonTests(unittest.TestCase):
    def test_eight_profiles_preserve_algorithm_and_other_hyperparameters(self):
        controls={j['task']:j for s in (0,1) for j in reference('/unused','test',s)['jobs']
                  if j['dacer_target_entropy_per_dim']==.7}
        jobs=[]
        for shard in (0,1):
            m=campaign_manifest('/unused','test',shard)
            self.assertEqual(len(m['jobs']),4)
            self.assertEqual(m['expected_regulator_updates'],16)
            self.assertEqual(m['wandb_project'],'antmaze')
            self.assertEqual(m['excluded_hosts'],['vast1'])
            jobs.extend(m['jobs'])
        self.assertEqual(len({j['id'] for j in jobs}),8)
        self.assertEqual({(j['task'],j['hypothesis']) for j in jobs},
            {(t,c) for t in ('v3','v4') for c in CONDITIONS}|{('v1','gamma999'),('v1','gamma999_temp3')})
        self.assertEqual((TOTAL_STEPS,expected_updates(TOTAL_STEPS)),(258304,7816))
        for j in jobs:
            self.assertEqual((j['discount'],j['temperature']),CONDITIONS[j['hypothesis']])
            self.assertEqual(j['steps'],258304)
            self.assertEqual(j['eval_interval'],50000)
            old=controls[j['task']];comparison=dict(j)
            for key in ('discount','hypothesis'):comparison.pop(key)
            for key in ('id','temperature','steps','eval_interval'):comparison[key]=old[key]
            self.assertEqual(comparison,old)

    def test_pending_assignment_priority_is_not_all_jobs_completion_barrier(self):
        self.assertTrue(priority_queue_ready({}))
        with tempfile.TemporaryDirectory() as folder:
            m={'priority_campaign':folder};p=Path(folder)/'status.json'
            self.assertFalse(priority_queue_ready(m))
            for state,want in [({},False),({'pending':['old'],'running':[]},False),
                ({'pending':[],'running':[{'pid':123}]},True),
                ({'pending':[],'running':[],'completed':['old']},True),
                ({'pending':[],'failed':['old']},False),
                ({'pending':[],'pending_held':True},False)]:
                p.write_text(json.dumps(state))
                self.assertEqual(priority_queue_ready(m),want)


if __name__=='__main__':unittest.main()
