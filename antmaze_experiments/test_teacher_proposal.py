"""Verify the bounded ablation isolates the existing teacher-only parameter."""
import unittest
from .teacher_proposal import validate_floor
from .register_teacher_floor_screen import campaign_manifest
from .register_start_normalized_geodesic import campaign_manifest as control_manifest
from .settings import expected_updates


class TeacherFloorTests(unittest.TestCase):
    def test_two_jobs_match_archived_control_except_floor_and_identity(self):
        original=control_manifest('/unused','test')
        control=next(j for j in original['jobs'] if j['temperature']==3.)
        new=campaign_manifest('/unused','test')
        self.assertEqual(len(new['jobs']),2)
        self.assertEqual({j['teacher_std_floor'] for j in new['jobs']},{.5,1.})
        self.assertEqual(new['excluded_hosts'],['vast1'])
        self.assertEqual(new['wandb_project'],'antmaze')
        for job in new['jobs']:
            comparable=dict(job)
            comparable.pop('teacher_std_floor')
            for key in ('id','hypothesis'):comparable[key]=control[key]
            self.assertEqual(comparable,control)
            self.assertEqual((job['steps'],expected_updates(job['steps'])),(258304,7816))
            self.assertEqual((job['eval_interval'],job['interim_eval_episodes'],job['final_eval_episodes']),
                             (50000,40,100))
        self.assertEqual(control_manifest('/unused','test'),original)

    def test_rejects_undefined_proposal_scale(self):
        for value in (0,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):validate_floor(value)
        for value in (.006737946999085467,.5,1.):
            self.assertEqual(validate_floor(value),value)


if __name__=='__main__':unittest.main()
