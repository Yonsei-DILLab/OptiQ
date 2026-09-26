import numpy as np
import pytest

from pointmaze.batch import TaskBatch
from pointmaze.drac_paper import PaperPointMaze, HORIZONS, GOAL_COUNTS
from pointmaze.run import configuration, expected_updates
from pointmaze.replay import Replay


def test_paper_settings():
    cfg = configuration()
    assert expected_updates() == 62000
    assert cfg.alg.batch_size == 4096
    assert cfg.alg.actor.temperature == 5
    assert cfg.alg.actor.num_policy_samples == 64
    assert cfg.alg.actor.hidden_dims == [256, 256]
    assert cfg.alg.actor.log_std_min == -5
    assert cfg.alg.actor.log_std_max == cfg.alg.actor.initial_log_std == -1
    assert cfg.alg.actor.density_correction_beta == 1
    assert not cfg.dacer.enabled
    with pytest.raises(ValueError):
        expected_updates(1_000_000)


@pytest.mark.parametrize('maze', HORIZONS)
def test_maps(maze):
    plain, obstacle = PaperPointMaze(maze), PaperPointMaze(maze, True)
    try:
        obs, _ = plain.reset(seed=0)
        assert obs.shape == (4,)
        assert plain.action_space.shape == (2,)
        assert len(plain.goal_positions) == GOAL_COUNTS[maze]
        assert plain.horizon == HORIZONS[maze]
        assert obstacle.native.point_env.model.ngeom > plain.native.point_env.model.ngeom
        _, reward, _, _, _ = plain.step(np.zeros(2))
        assert reward == 0
        goal = plain.goal_positions[0]
        assert plain.native.compute_reward(obs[:2], goal) == 100
    finally:
        plain.close()
        obstacle.close()


def test_batch_and_replay():
    batch = TaskBatch('simple', 2, 0)
    try:
        obs = batch.current.copy()
        actions = np.zeros((2, 2), np.float32)
        next_obs, rewards, terminated, truncated, _ = batch.step(actions)
        replay = Replay(capacity=4)
        replay.add(obs, actions, rewards, next_obs, terminated | truncated)
        assert replay.size == 2
        np.testing.assert_array_equal(replay.data['next_observations'][:2], next_obs)
        np.testing.assert_array_equal(replay.data['actions'][:2], actions)
    finally:
        batch.close()
