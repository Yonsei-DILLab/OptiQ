import numpy as np

from scripts.diagnose_v2_soft_critic import discounted_q_samples


def test_soft_q_return_excludes_current_entropy_and_ends_at_terminal_reward():
    rewards = np.array([2., 3., 4.])
    entropy = np.array([1000., 2., -1.])
    actual = discounted_q_samples(rewards, entropy, .5, .9)
    expected = np.array([2. + .9 * (3. + .9 * (4. - .5) + 1.), 3. + .9 * (4. - .5), 4.])
    np.testing.assert_allclose(actual, expected)
    entropy[0] = -1000.
    np.testing.assert_allclose(discounted_q_samples(rewards, entropy, .5, .9), expected)
    np.testing.assert_allclose(discounted_q_samples(rewards, entropy, 0., .9), [7.94, 6.6, 4.])
