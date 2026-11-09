import numpy as np
import json
import pytest

import pointmaze_rebuttal.reporting as reporting
from pointmaze_rebuttal.reporting import aggregate, discovery_metrics, goal_metrics


def test_discovery_tracks_real_rewards_new_goals_and_collection_time():
    rewards = np.zeros(20, np.float32)
    rewards[[5, 6, 13]] = 100
    states = np.zeros((20, 4), np.float32)
    goals = np.array([[0, 0], [3, 0], [0, 3], [3, 3]], np.float32)
    states[13, :2] = goals[2]
    result = discovery_metrics(rewards, states, goals, num_envs=4, warmup=4)
    assert result['first_reward_at_collection_end'] == 8
    assert result['first_goal_reward_at_collection_end'] == [8, None, 16, None]
    assert result['training_goal_counts'] == [2, 0, 1, 0]
    assert result['training_goal_coverage'] == 2
    assert result['warmup_reward_events'] == 0
    assert result['all_goals_discovered'] is False
    assert result['curve']['goals_discovered'][-1] == 2


def test_undiscovered_seed_stays_in_discovery_denominator():
    goals = np.array([[0, 0], [3, 0], [0, 3], [3, 3]], np.float32)
    entries = []
    for seed in range(5):
        reward = np.zeros(20, np.float32)
        if seed < 2:
            reward[5] = 100
        train = discovery_metrics(reward, np.zeros((20, 4)), goals, 4, 4)
        entries.append(dict(method='sac', seed=seed, total_steps=20, training=train, evaluations=[]))
    row = next(r for r in aggregate(entries) if r['method'] == 'sac')
    assert row['reward_discovery_seeds'] == 2
    assert row['reward_not_discovered_seeds'] == 3
    assert row['discovery_curve'][-1]['discovered_seed_fraction'] == .4
    assert next(p for p in row['discovery_curve'] if p['transitions'] == 8)['discovered_seed_fraction'] == .4
    assert next(p for p in row['discovery_curve'] if p['transitions'] == 4)['discovered_seed_fraction'] == 0.
    assert row['first_reward_conditional_on_discovery']['n'] == 2
    assert row['complete'] is False


def test_goal_coverage_is_within_seed_and_entropy_is_undefined_without_success():
    result = goal_metrics(dict(goals=[1, 0, 0, 0], failure=4, episodes=5, reachable_goals=1, success=.2))
    assert result['normalized_goal_entropy'] == 0.
    assert result['effective_goals'] == 1.
    empty = goal_metrics(dict(goals=[0, 0, 0, 0], failure=5, episodes=5, reachable_goals=0, success=0.))
    assert empty['normalized_goal_entropy'] is None
    assert empty['effective_goals'] is None


@pytest.mark.parametrize('dipo_keys', [False, True])
def test_wrapped_replay_recovers_prefix_and_chronology_for_both_formats(tmp_path, monkeypatch, dipo_keys):
    total, capacity, num_envs = 20, 18, 4
    goal = np.array([[0, 0], [3, 0], [0, 3], [3, 3]], np.float32)
    reward = np.zeros(total, np.float32)
    reward[[5, 13]] = 100
    following = np.zeros((total, 4), np.float32)
    following[13, :2] = goal[2]
    complete = dict(observations=np.zeros((total, 4), np.float32),
                    actions=np.zeros((total, 2), np.float32), rewards=reward,
                    next_observations=following)
    prefix = {k: v[:num_envs] for k, v in complete.items()}
    prefix['goals'] = goal
    monkeypatch.setattr(reporting, 'initial_collection', lambda config, cache: prefix)
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'config.json').write_text(json.dumps(dict(steps=total, num_envs=num_envs,
        warmup=4, replay_capacity=capacity, expected_updates=16)))
    (tmp_path / 'completed.json').write_text(json.dumps(dict(completed=True, updates=16)))
    stored = {}
    for key, values in complete.items():
        target = np.empty((capacity,) + values.shape[1:], np.float32)
        for i, value in enumerate(values):
            target[i % capacity] = value
        name = {'observations': 'states', 'next_observations': 'next_states'}.get(key, key) if dipo_keys else key
        stored[name] = target[:, None] if key == 'rewards' and dipo_keys else target
    np.savez_compressed(tmp_path / 'checkpoints/final-replay.npz', **stored, size=capacity, position=2)
    result = reporting.analyze_replay(tmp_path, tmp_path / 'cache')
    assert result['prefix_reconstructed_transitions'] == 2
    assert result['first_reward_at_collection_end'] == 8
    assert result['first_goal_reward_at_collection_end'] == [8, None, 16, None]
    assert result['training_goal_counts'] == [1, 0, 1, 0]
    assert result['reward_events'] == 2
