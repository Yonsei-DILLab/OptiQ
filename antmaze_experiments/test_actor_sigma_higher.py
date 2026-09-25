"""Guard default preservation and exact single-parameter experiment scope."""
import unittest
import math
from .actor_sigma_profile import settings
from .teacher_proposal import verify_update_summary
from .register_actor_sigma_higher import campaign_manifest, HOSTS
from .register_teacher_floor_screen import campaign_manifest as v3_manifest
from .register_v4_teacher_floor import campaign_manifest as v4_manifest


class HigherSigmaTests(unittest.TestCase):
    def test_actual_bound_validation_does_not_silently_relax_defaults(self):
        info={'train/actor_std_mean':.8}
        with self.assertRaises(AssertionError):verify_update_summary(.5,info,.5)
        verify_update_summary(.5,info,.5,(-5.,0.))
        verify_update_summary(.5,{'train/actor_std_mean':20.},.5,(-5.,3.))
        verify_update_summary(.5,{'train/actor_std_mean':100.},.5,(-5.,float('inf')))
        with self.assertRaises(AssertionError):
            verify_update_summary(.5,{'train/actor_std_mean':21.},.5,(-5.,3.))
        with self.assertRaises(AssertionError):
            verify_update_summary(.5,{'train/actor_std_mean':float('inf')},.5,(-5.,float('inf')))
        with self.assertRaises(AssertionError):
            verify_update_summary(.5,info,.5,(-5.,float('nan')))
        self.assertEqual(settings(),dict(log_std_min=-5.,log_std_max=-1.,initial_log_std=-1.))

    def test_only_upper_bound_changes_relative_to_controls(self):
        controls=[next(j for j in v3_manifest('/unused','test')['jobs'] if j['teacher_std_floor']==1.),
                  next(j for j in v4_manifest('/unused','test')['jobs'] if j['teacher_std_floor']==.5)]
        manifests=[campaign_manifest('/unused','test',host) for host in HOSTS]
        jobs=[job for m in manifests for job in m['jobs']]
        self.assertEqual(len(jobs),10)
        self.assertEqual(len({j['id'] for j in jobs}),10)
        for host,m in zip(HOSTS,manifests):
            self.assertEqual(m['host'],host)
            self.assertEqual(len(m['jobs']),5)
            self.assertIsNone(m['priority_campaign'])
        for task in ('v3','v4'):
            self.assertEqual({settings(j['actor_sigma_profile'])['log_std_max'] for j in jobs if j['task']==task},
                             {0.,1.,2.,3.,math.inf})
        for job in jobs:
            control=next(j for j in controls if j['task']==job['task'])
            self.assertEqual({k for k in set(job)|set(control) if job.get(k)!=control.get(k)},
                             {'id','hypothesis','actor_sigma_profile'})
            profile=settings(job['actor_sigma_profile'])
            self.assertEqual({k for k in profile if profile[k]!=settings()[k]},{'log_std_max'})
            self.assertIn(profile['log_std_max'],(0.,1.,2.,3.,math.inf))
            self.assertEqual(job['steps'],258304)
            self.assertEqual((job['interim_eval_episodes'],job['final_eval_episodes']),(40,100))


if __name__=='__main__':unittest.main()
