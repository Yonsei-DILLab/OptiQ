import unittest

from .run import dense_reward_specification, route_summary
from .register_fixed_goal_dense import campaign_manifest, TOTAL_STEPS, GPU


class FixedGoalDenseTests(unittest.TestCase):
    def test_dense_reward_keeps_both_fixed_goals(self):
        v3 = dense_reward_specification('v3')
        v4 = dense_reward_specification('v4')
        self.assertEqual(v3['formula'], 'r_t = -min_g ||p_{t+1} - g||_2')
        self.assertEqual(v3['goal_count'], 2)
        self.assertEqual(v3['goals'], [[-12.0, 12.0], [12.0, -12.0]])
        self.assertEqual(v4['goal_count'], 2)
        self.assertEqual(v4['goals'], [[-16.0, 4.0], [-16.0, -4.0]])
        self.assertIn('no random goal selection', v3['goal_set'])
        self.assertIn('no random goal selection', v4['goal_set'])

    def test_route_counts_and_proportions_v3(self):
        paths = [
            [[0, 0], [-9, 0]],
            [[0, 0], [9, 0]],
            [[0, 0], [-9, 0], [9, 0]],
            [[0, 0], [2, 0]],
        ]
        result = route_summary('v3', paths, [1, 0, 1, 0])
        self.assertEqual(result['route_counts'],
                         {'left': 1, 'right': 1, 'both': 1, 'uncommitted': 1})
        self.assertEqual(result['route_proportions'],
                         {'left': .25, 'right': .25, 'both': .25, 'uncommitted': .25})
        self.assertEqual(result['route_success_counts'],
                         {'left': 1, 'right': 0, 'both': 1, 'uncommitted': 0})

    def test_route_counts_and_proportions_v4(self):
        paths = [[[0, 0], [-5, 3]], [[0, 0], [-5, -3]], [[0, 0], [-2, 0]]]
        result = route_summary('v4', paths, [1, 0, 0])
        self.assertEqual(result['route_counts'],
                         {'upper': 1, 'lower': 1, 'uncommitted': 1})
        self.assertAlmostEqual(sum(result['route_proportions'].values()), 1.)

    def test_campaign_is_limited_to_two_5090_gpu3_slots(self):
        for shard, task, host in ((0, 'v3', 'vast-heechan-180'),
                                  (1, 'v4', 'vast-heechan-199')):
            manifest = campaign_manifest('/frozen/source', 'commit', shard)
            self.assertEqual(manifest['host'], host)
            self.assertEqual(manifest['eligible_gpus'], [GPU])
            self.assertEqual(manifest['jobs'][0]['task'], task)
            self.assertEqual(manifest['jobs'][0]['train_starts'], 'fixed')
            self.assertEqual(manifest['jobs'][0]['eval_starts'], 'upstream')
            self.assertEqual(manifest['jobs'][0]['steps'], TOTAL_STEPS)
            self.assertLessEqual(TOTAL_STEPS, 1_000_000)
            self.assertEqual(manifest['jobs'][0]['reward_profile'], 'dense')
            self.assertEqual(manifest['jobs'][0]['fixed_goal_coordinates'],
                             dense_reward_specification(task)['goals'])
            self.assertEqual(manifest['jobs'][0]['default_nm'], 64)


if __name__ == '__main__':
    unittest.main()
