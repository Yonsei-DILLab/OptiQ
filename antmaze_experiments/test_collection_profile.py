"""Accounting regressions for optional collection/update schedules."""
import unittest

from .collection_profile import get_profile, expected_updates, aligned_eval_step
from .settings import expected_updates as legacy_updates, WARMUP


class CollectionAccounting(unittest.TestCase):
    def test_default_accounting_matches_legacy_at_every_block(self):
        self.assertEqual(get_profile(), dict(num_envs=256, updates_per_vector_step=8))
        for step in range(WARMUP, 1008384 + 1, 256):
            self.assertEqual(expected_updates(step), legacy_updates(step))

    def test_same_budget_and_update_count(self):
        self.assertEqual(get_profile('env32-update1'), dict(num_envs=32, updates_per_vector_step=1))
        for total, count in ((8448, 8), (258304, 7816), (1008384, 31256)):
            self.assertEqual(expected_updates(total, 'env32-update1'), count)
            self.assertEqual(expected_updates(total, 'env32-update1'), legacy_updates(total))
        for i, step in enumerate(range(WARMUP+32, 258304+1, 32), 1):
            self.assertEqual(expected_updates(step, 'env32-update1'), i)

    def test_256_updates_per_256_new_transitions(self):
        profile = get_profile('env256-update256')
        self.assertEqual(profile, dict(num_envs=256, updates_per_vector_step=256))
        self.assertEqual(expected_updates(WARMUP, 'env256-update256'), 0)
        self.assertEqual(expected_updates(WARMUP+256, 'env256-update256'), 256)
        self.assertEqual(expected_updates(WARMUP+512, 'env256-update256'), 512)
        self.assertEqual(expected_updates(WARMUP+512), 16)
        with self.assertRaises(ValueError):
            expected_updates(WARMUP+257, 'env256-update256')

    def test_identical_evaluation_checkpoints(self):
        for interval in (50000, 250000):
            total = 1008384
            legacy, cutoff = [], interval
            for step in range(256, total+1, 256):
                if step >= cutoff and step < total:
                    legacy.append(step)
                    cutoff += interval
            smaller, index = [], 1
            for step in range(32, total+1, 32):
                if step >= aligned_eval_step(index, interval) and step < total:
                    smaller.append(step)
                    index += 1
            self.assertEqual(smaller, legacy)
        self.assertEqual([aligned_eval_step(i, 50000) for i in range(1, 6)],
                         [50176, 100096, 150016, 200192, 250112])

    def test_invalid_counts_do_not_silently_round(self):
        for step in (WARMUP-32, WARMUP+1, WARMUP+31):
            with self.assertRaises(ValueError):
                expected_updates(step, 'env32-update1')
        with self.assertRaises(ValueError):
            get_profile('env64')


if __name__ == '__main__':
    unittest.main()
