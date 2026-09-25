import unittest

from .register_utd256_random_starts import campaign_manifest
from .register_utd256 import campaign_manifest as control_manifest


class RandomStartsV34Tests(unittest.TestCase):
    def test_only_start_distribution_and_identity_change(self):
        for host, task, shard in (('vast-heechan-180', 'v3', 0),
                                  ('vast-heechan-199', 'v4', 1)):
            new = campaign_manifest('/unused', 'test', host)
            old = control_manifest('/unused', 'test', shard, 'basic_euclidean')
            job = dict(new['jobs'][0])
            previous = next(j for j in old['jobs'] if j['task'] == task)
            self.assertEqual(job.pop('train_starts'), 'random')
            self.assertEqual(job.pop('eval_starts'), 'random')
            self.assertEqual(previous.pop('eval_starts'), 'upstream')
            self.assertNotEqual(job.pop('id'), previous.pop('id'))
            self.assertEqual(job, previous)
            self.assertEqual(new['wandb_project'], 'antmaze')
            self.assertEqual(new['jobs'][0]['temperature'], 1.)
            self.assertEqual(new['jobs'][0]['steps'],
                             old['jobs'][0]['steps'] if old['jobs'][0]['task'] == task else
                             next(j['steps'] for j in old['jobs'] if j['task'] == task))


if __name__ == '__main__':
    unittest.main()
