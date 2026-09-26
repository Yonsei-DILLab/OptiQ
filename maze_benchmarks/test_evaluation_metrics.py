"""Check removal robustness against exhaustive subset enumeration."""
import itertools
import unittest

import numpy as np

from .evaluation_metrics import removal_sr5_exact


class RemovalRobustnessTest(unittest.TestCase):
    def test_every_possible_reached_cardinality(self):
        for count in (4, 8):
            for distinct in range(min(count, 5) + 1):
                ids = list(range(distinct)) + [-1] * (5 - distinct)
                outcomes = [any(g >= 0 and g not in removed for g in ids)
                            for removed in itertools.combinations(range(count), count // 2)]
                with self.subTest(goals=count, distinct=distinct):
                    self.assertAlmostEqual(removal_sr5_exact(ids, count), np.mean(outcomes))

    def test_exactly_half_is_not_certain_success(self):
        self.assertAlmostEqual(removal_sr5_exact([0, 1, 0, 1, -1], 4), 5 / 6)
        self.assertAlmostEqual(removal_sr5_exact([0, 1, 2, 3, -1], 8), 69 / 70)

    def test_failed_and_successful_groups_are_all_counted(self):
        self.assertAlmostEqual(removal_sr5_exact([-1] * 5 + [0] * 5, 4), .25)

    def test_invalid_goal_ids_rejected(self):
        for ids in ([0, 1, 2, 3, 4], [-2, 0, 1, 2, 3], [0, 1]):
            with self.assertRaises(ValueError):
                removal_sr5_exact(ids, 4)


if __name__ == "__main__":
    unittest.main()
