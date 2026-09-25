"""Regression guards for default preservation and isolation of sigma ablations."""
from pathlib import Path
import unittest

from .actor_sigma_profile import settings
from .register_actor_sigma_screen import campaign_manifest, INITIAL
from .register_teacher_floor_screen import campaign_manifest as v3_manifest
from .register_v4_teacher_floor import campaign_manifest as v4_manifest


class SigmaProfileTest(unittest.TestCase):
    def test_default_unchanged_and_not_mutable_through_result(self):
        result = settings()
        result['log_std_max'] = 20
        self.assertEqual(settings(), dict(log_std_min=-5.,log_std_max=-1.,initial_log_std=-1.))
        with self.assertRaises(KeyError):
            settings('unbounded')

    def test_only_cap_and_init_are_forwarded(self):
        for name, upper in [('capm2',-2.), ('capm3',-3.)]:
            self.assertEqual(settings(name),dict(log_std_min=-5.,log_std_max=upper,initial_log_std=upper))

    def test_manifest_matches_actual_control_definitions(self):
        source = Path('/tmp/nonexecuting-manifest-test')
        controls = [next(j for j in v3_manifest(source,'test')['jobs'] if j['teacher_std_floor']==1.),
                    next(j for j in v4_manifest(source,'test')['jobs'] if j['teacher_std_floor']==.5)]
        manifest = campaign_manifest(source,'test')
        self.assertEqual(len(manifest['jobs']),4)
        self.assertEqual(manifest['initial_parameter_control'],INITIAL)
        self.assertEqual(manifest['num_envs'],256)
        for job in manifest['jobs']:
            control = next(j for j in controls if j['task']==job['task'])
            diff = {k for k in set(job)|set(control) if job.get(k)!=control.get(k)}
            self.assertEqual(diff,{'id','hypothesis','actor_sigma_profile'})
            self.assertNotIn('collection_profile',job)
            self.assertNotIn('latent_profile',job)


if __name__ == '__main__':
    unittest.main()
