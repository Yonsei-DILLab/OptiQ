import numpy as np

from environment import FourWayEnv


def test_fourway_goals_and_rewards_are_symmetric():
    env = FourWayEnv()
    expected = np.asarray([[5., 0.], [-5., 0.], [0., 5.], [0., -5.]])
    np.testing.assert_array_equal(env.goal_positions, expected)
    for action, expected_goal in (([1., 0.], 0), ([-1., 0.], 1),
                                  ([0., 1.], 2), ([0., -1.], 3)):
        obs, _ = env.reset(seed=0)
        nxt, reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert info["nearest_goal_id"] == expected_goal
        assert reward == -30.0 - 16.0
        np.testing.assert_array_equal(nxt, np.asarray(action, np.float32))
    env.close()
