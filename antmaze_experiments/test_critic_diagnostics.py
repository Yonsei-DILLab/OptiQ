"""Pure-math and recorder tests; no MuJoCo, training or GPU required."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from . import critic_diagnostics as cd


class CriticDiagnosticsTests(unittest.TestCase):
    def test_signed_error_cancellation_does_not_hide_magnitude(self):
        result = cd._stats([-2., 2.])
        self.assertEqual(result['mean'], 0.)
        self.assertEqual(result['mean_abs'], 2.)
        self.assertEqual(result['rms'], 2.)
        self.assertIsNone(cd._stats([])['rms'])

    def test_terminal_and_residual_identity(self):
        terms = cd.decompose([1., 2.], [False, True], [2., 3.],
                             [3., 999.], [2., -100.], [1.5, -200.], .9)
        np.testing.assert_allclose(terms['online_bellman_residual'], [1.7, -1.])
        self.assertEqual(terms['target_tracking_term'][1], 0.)
        self.assertEqual(terms['target_twin_min_term'][1], 0.)
        total = sum(np.dot([1., .9], v) for k, v in terms.items() if k != 'online_bellman_residual')
        self.assertAlmostEqual(total, cd.discounted_returns([1., 2.], .9)[0] - 2.)

    def test_timeout_tail_and_reward_units(self):
        raw = cd.discounted_returns([1., 2.], .9)
        boot = cd.discounted_returns([1., 2.], .9, 4.)
        self.assertAlmostEqual(boot[0] - raw[0], .9 ** 2 * 4.)
        np.testing.assert_allclose(cd.discounted_returns([.2, .4], .9, .8), .2 * boot)

    def test_route_labels_use_trajectory(self):
        self.assertEqual(cd.route_label('v3', [[0, 0], [-9, 0], [9, 0]]), ('both', 0))
        self.assertEqual(cd.route_label('v4', [[0, 0], [-3, 3], [-5, 3]]), ('upper', 1))
        self.assertEqual(cd.route_label('v4', [[0, 0], [-3, -3], [-5, -3]]), ('lower', 1))
        self.assertEqual(cd.route_label('v4', [[0, 0], [-1, -3]]), ('uncommitted', -1))

    def test_recorder_excludes_inactive_slots_and_autoreset(self):
        recorder = cd.EvaluationTraceRecorder()
        recorder.start_batch([True, False])
        recorder.record([[0, 0], [88, 88]], [[.1], [.2]], [1., 999.],
                        [[1, 0], [99, 99]], [True, False], [True, False], [True, False])
        recorder.end_batch()
        self.assertEqual(len(recorder.episodes), 1)
        np.testing.assert_array_equal(recorder.episodes[0]['next_observations'][0], [1, 0])
        with self.assertRaises(RuntimeError):
            recorder.record([], [], [], [], [], [], [])

    def test_full_summary_normalization_and_telescoping(self):
        for terminal in (False, True):
            recorder = cd.EvaluationTraceRecorder()
            recorder.start_batch([True])
            recorder.record([[0., 0.]], [[0.]], [1.], [[1., 0.]], [False], [False], [True])
            recorder.record([[1., 0.]], [[0.]], [2.], [[2., 0.]], [True], [terminal], [True])
            recorder.end_batch()
            cfg = SimpleNamespace(alg=SimpleNamespace(critic=SimpleNamespace(n_atoms=1)))
            learner = SimpleNamespace(method='optiq', model=SimpleNamespace(policy=object(), gamma=.9, cfg=cfg))
            def q_values(p, obs, actions):
                x = np.asarray(obs)[:, 0]
                return np.stack([x+2.2, x+1.8], -1), np.stack([x+1.5, x+.5], -1)
            def boundary(p, obs, seed, draws):
                x = np.asarray(obs)[:, 0]
                return dict(online_mean=x+2., target_mean=x+1., target_min=x+.5)
            with tempfile.TemporaryDirectory() as tmp, patch.object(cd, '_guard', return_value=('stable',)), \
                    patch.object(cd, 'evaluate_q', side_effect=q_values), patch.object(cd, 'policy_values', side_effect=boundary), \
                    patch.object(cd, 'teacher_probe', return_value=({}, {'states': 1})):
                result = recorder.save(learner, 'v3', tmp, 100, reward_multiplier=.2, goals=[0])
                record = result['routes']['uncommitted']['landmarks']['0']
                self.assertAlmostEqual(record['raw_mc']['raw']['mean'], 2.8)
                self.assertAlmostEqual(record['raw_mc']['normalized']['mean'], 14.)
                self.assertAlmostEqual(record['bootstrap_mc']['raw']['mean'], 2.8 if terminal else 6.04)
                raw = np.load(Path(tmp)/'critic-trace.npz')
                self.assertEqual(raw['q_online'].shape, (2, 2))
                np.testing.assert_array_equal(raw['terminals'], [False, terminal])
                self.assertTrue(json.loads((Path(tmp)/'critic-diagnostics.json').read_text())['verification']['residual_identity_verified'])


if __name__ == '__main__':
    unittest.main()
