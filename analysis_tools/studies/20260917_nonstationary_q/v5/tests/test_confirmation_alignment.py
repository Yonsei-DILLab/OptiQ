import numpy as np

from scripts.analyze_v2_confirmation import aligned_pair


def test_asynchronous_latest_evaluations_do_not_create_a_false_win():
    pair = aligned_pair(np.array([100, 200, 300]), np.array([1., 4., 99.]),
                        np.array([100, 150, 200]), np.array([2., 1000., 8.]), tail=2)
    np.testing.assert_array_equal(pair["steps"], [100, 200])
    assert pair["step"] == 200
    assert pair["v2_tail_mean"] == 2.5 and pair["optiq_tail_mean"] == 5.
    assert pair["ratio"] == .5


def test_stopped_seed_keeps_its_last_common_step():
    pair = aligned_pair(np.array([100, 200]), np.array([4., 3.]),
                        np.array([100, 200, 300, 400]), np.array([4., 5., 9., 12.]))
    assert pair["step"] == 200
    assert pair["ratio"] == 3.5 / 4.5
