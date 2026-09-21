import unittest
from common import ROOT, read, command
from submit import jobs


class PlanTests(unittest.TestCase):
    def test_sizes_and_validation_isolation(self):
        p = read(ROOT / 'plan.json')
        args = command(p, 'humanoid', 3, 'a' * 40)
        for value in ['alg.actor.num_policy_samples=128', 'alg.actor.proposals_per_policy_sample=2',
                      'alg.actor.temperature=0.25', 'total_steps=1000000', 'wandb.mode=online']:
            self.assertIn(value, args)
        short = command(p, 'humanoid', 0, 'a' * 40, True)
        self.assertIn('total_steps=128', short)
        self.assertTrue(any('/validation/' in a for a in short))
        self.assertTrue(any(a.endswith('_validation') for a in short))

    def test_dependency_chain(self):
        p = read(ROOT / 'plan.json')
        iterator = jobs(p)
        stage, args = next(iterator)
        self.assertEqual(stage, 'validate')
        self.assertFalse(any('--dependency' in a for a in args))
        for index, expected in enumerate(['ant', 'humanoid', 'hopper', 'walker2d', 'halfcheetah']):
            stage, args = iterator.send(str(100 + index))
            self.assertEqual(stage, expected)
            self.assertIn('--dependency=afterok:' + str(100 + index), args)
            self.assertIn('--array=0-3%4', args)
        with self.assertRaises(StopIteration):
            iterator.send('105')


if __name__ == '__main__':
    unittest.main()
