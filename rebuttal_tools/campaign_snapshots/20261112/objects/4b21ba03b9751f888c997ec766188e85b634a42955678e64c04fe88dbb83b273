"""Verify real task trajectories and task-specific HER before releasing cohorts."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json
import pickle
import numpy as np
from gymnasium import spaces
from rebuttal_tools.fetch_suite_vector_replay import VectorHERReplayBuffer, NativeReplayView


def check_task(task):
    from fetch_rebuttal.env import make_env
    from rebuttal_tools.fetch_suite_vector_env import GroupedFetchVecEnv
    vec = GroupedFetchVecEnv(4, [[],[]], task)
    scalar = [make_env(task) for _ in range(4)]
    try:
        vec.seed(91700)
        assert np.array_equal(vec.reset(), np.stack([e.reset(seed=91700+i)[0] for i,e in enumerate(scalar)]))
        for step in range(52):
            actions = np.tile(np.array([.1,-.2,.05,-.3], np.float32), (4,1))
            obs, rewards, dones, infos = vec.step(actions)
            reference = [e.step(a) for e,a in zip(scalar,actions)]
            assert np.array_equal(rewards, [r[1] for r in reference])
            assert np.array_equal(np.stack([i['fetch_raw_next_observation'] for i in infos]), np.stack([r[0] for r in reference]))
            if step == 49:
                assert dones.all() and all(i['TimeLimit.truncated'] for i in infos)
                for e in scalar:
                    e.reset()
            else:
                assert np.array_equal(obs, np.stack([r[0] for r in reference]))
        dim = vec.observation_space.shape[0]
        replay = VectorHERReplayBuffer(28, vec.observation_space, vec.action_space,
            n_envs=2, device='cpu', task=task, compute_reward=scalar[0].unwrapped.compute_reward, her_seed=42)
        offset = 0 if task == 'reach' else 3
        def state(t):
            x = np.zeros((2,dim), np.float32)
            x[:,offset] = np.arange(2)*100+t
            x[:,offset+1] = t
            x[:,offset+2] = .4
            x[:,-3:] = -20
            return x
        for t in range(40):
            nxt = state(t+1)
            done = np.array([(t+1)%3 == 0, (t+1)%4 == 0])
            infos = [{'fetch_raw_next_observation':nxt[i], 'TimeLimit.truncated':bool(done[i])} for i in range(2)]
            replay.add(state(t), np.zeros_like(nxt), np.zeros((2,4)), np.full(2,-1.), done, infos)
        before = replay.observations.copy()
        class Normalizer:
            def normalize_obs(self, x): return x+10000
            def normalize_reward(self, r): return r
        samples = replay.sample(8192, Normalizer())
        o = samples.observations.numpy()-10000
        n = samples.next_observations.numpy()-10000
        her = np.arange(8192)%256 < 204
        assert np.all(o[her,-3]//100 == o[her,offset]//100), 'Cross-environment HER goal'
        assert np.all(o[her,-2] >= n[her,offset+1]), 'Goal is not a future achieved state'
        assert np.all((o[:,-3:] != -20).any(1).reshape(-1,256).sum(1) == 204)
        expected = scalar[0].unwrapped.compute_reward(n[her,offset:offset+3], n[her,-3:], None)
        assert np.array_equal(samples.rewards.numpy()[her,0], expected)
        assert np.all(samples.dones.numpy() == 0), 'Timeout incorrectly terminates target'
        assert np.array_equal(before, replay.observations)
        loaded = pickle.loads(pickle.dumps(replay))
        assert loaded.task == task and loaded.compute_reward is None
        assert np.array_equal(loaded.episode_ends, replay.episode_ends)
        batch = NativeReplayView(replay, Normalizer()).batch(256)
        assert batch['observations'].shape == (256,dim) and batch['terminals'].shape == (256,)
        return dict(passed=True, task=task, observation_dim=dim, achieved_goal_slice=[offset,offset+3],
            checks=['Scalar/vector paired trajectories and autoresets', 'Ring overwrite and episode boundaries',
                '204 HER samples per optimizer minibatch', 'Task-specific sparse reward before normalization',
                'Timeout bootstrapping', 'Raw replay unchanged', 'Replay serialization', 'Native minibatch interface'])
    finally:
        vec.close()
        for e in scalar:
            e.close()


if __name__ == '__main__':
    print(json.dumps([check_task(task) for task in ['push','slide','reach','pick-and-place']]), flush=True)
