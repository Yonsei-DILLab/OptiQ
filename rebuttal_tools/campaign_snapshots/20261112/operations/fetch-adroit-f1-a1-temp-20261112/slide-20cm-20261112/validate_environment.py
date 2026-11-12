"""Validate Slide 20cm physics, collection, HER, and evaluation on each runtime."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import sys
import time


def main():
    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np
    from fetch_rebuttal.env import make_env
    from fetch_rebuttal.run import evaluate
    from rebuttal_tools.fetch_suite_vector_env import GroupedFetchVecEnv
    from rebuttal_tools.fetch_suite_vector_replay import VectorHERReplayBuffer

    base = Path(__file__).resolve().parent
    report = dict(at=time.time(), distance_threshold_m=.20, horizon=50,
                  physics_and_goal_distribution_unchanged=True)
    thresholds = {}
    for task in ['reach', 'push', 'pick-and-place', 'slide']:
        e = make_env(task)
        thresholds[task] = float(e.unwrapped.distance_threshold)
        assert e.spec.max_episode_steps == 50
        assert thresholds[task] == (.20 if task == 'slide' else .05)
        e.close()
    report['task_thresholds'] = thresholds

    env = make_env('slide')
    native = gym.make('FetchSlide-v4', reward_type='sparse')
    assert native.unwrapped.distance_threshold == .05
    obs, _ = env.reset(seed=92317)
    raw, _ = native.reset(seed=92317)
    assert np.array_equal(obs, np.concatenate([raw['observation'], raw['desired_goal']]).astype(np.float32))
    rng = np.random.default_rng(3188)
    trace = hashlib.sha256()
    for t in range(50):
        action = rng.uniform(-1, 1, 4).astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)
        old_obs, _, old_term, old_trunc, _ = native.step(action)
        expected = np.concatenate([old_obs['observation'], old_obs['desired_goal']]).astype(np.float32)
        assert np.array_equal(obs, expected)
        assert (term, trunc) == (old_term, old_trunc)
        distance = np.linalg.norm(old_obs['achieved_goal']-old_obs['desired_goal'])
        assert reward == -float(distance > .20)
        assert float(info['is_success']) == float(distance < .20)
        assert bool(term or trunc) == (t == 49)
        trace.update(obs.tobytes())
    report['physical_rollout_sha256'] = trace.hexdigest()
    distances = np.array([0., .049, .051, .199, .20, .201], dtype=np.float64)
    achieved = np.zeros((len(distances), 3)); achieved[:, 0] = distances
    desired = np.zeros_like(achieved)
    reward = env.unwrapped.compute_reward(achieved, desired, {})
    success = env.unwrapped._is_success(achieved, desired)
    assert np.array_equal(reward, -(distances > .20).astype(np.float32))
    assert np.array_equal(success, (distances < .20).astype(np.float32))
    report['boundary'] = dict(distances_m=distances.tolist(), sparse_rewards=reward.tolist(), success=success.tolist())

    vec = GroupedFetchVecEnv(256, [[] for _ in range(4)], task='slide')
    try:
        assert len(vec.processes) == 4
        values = vec.get_attr('distance_threshold')
        assert len(values) == 256 and all(v == .20 for v in values)
        replay = VectorHERReplayBuffer(256*60, vec.observation_space, vec.action_space,
            device='cpu', n_envs=256, compute_reward=env.unwrapped.compute_reward,
            her_seed=815, task='slide')
        vec.seed(3801); obs = vec.reset()
        for t in range(50):
            actions = np.zeros((256, 4), dtype=np.float32)
            nxt, rewards, dones, infos = vec.step(actions)
            raw_nxt = np.array([i['fetch_raw_next_observation'] for i in infos])
            distance = np.linalg.norm(raw_nxt[:, 3:6] - raw_nxt[:, -3:], axis=1)
            assert np.array_equal(rewards, -(distance > .20).astype(np.float32))
            assert all(bool(d) == (t == 49) for d in dones)
            if t == 49:
                assert all(i['TimeLimit.truncated'] for i in infos)
            replay.add(obs, nxt, actions, rewards, dones, infos)
            obs = nxt
        sampled = replay.sample(256)
        nxt = sampled.next_observations.numpy()
        expected = -(np.linalg.norm(nxt[:, 3:6] - nxt[:, -3:], axis=1) > .20).astype(np.float32)
        assert np.array_equal(sampled.rewards.numpy().reshape(-1), expected)
        assert replay.last['relabeled_samples'] == 204
        assert replay.last['sampled_future_within_episode'] and replay.last['sampled_future_same_environment']
        assert np.array_equal(sampled.dones.numpy(), np.zeros((256, 1)))
        report['collector_and_her'] = dict(num_envs=256, worker_processes=4, transitions=12800,
            all_thresholds_equal=True, full_episodes=256, relabeled_samples=204,
            sampled_reward_matches_20cm=True, timeout_bootstrapping_preserved=True)
    finally:
        vec.close(); env.close(); native.close()

    class ZeroPolicy:
        def act(self, obs, mode='policy'):
            return np.zeros((len(obs), 4), dtype=np.float32)

        @contextmanager
        def evaluation_rng(self, seed):
            yield

    evaluation = evaluate(ZeroPolicy(), 'slide', 10, 700305, 'stochastic_z')
    assert evaluation['distance_threshold_m'] == .20
    assert evaluation['lengths'] == [50]*10
    report['evaluation'] = evaluation
    report['passed'] = True
    (base/'environment-verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
