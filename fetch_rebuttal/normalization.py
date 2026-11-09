"""Common observation-only normalization for private Fetch experiments."""
from pathlib import Path
import json
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SETTINGS = dict(implementation='SB3 VecNormalize', norm_obs=True,
    norm_reward=False, clip_obs=5.0, epsilon=1e-8,
    statistics='online training observations only; frozen during evaluation',
    replay='raw observations; normalize both states using current statistics at sampling')


class RawTransition(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, fetch_raw_next_observation=obs.copy())
        return obs, reward, terminated, truncated, info


def make_normalized_env(task):
    from .env import make_env
    return VecNormalize(DummyVecEnv([lambda: RawTransition(make_env(task))]),
        norm_obs=True, norm_reward=False, clip_obs=5., epsilon=1e-8)


def save_normalizer(normalizer, folder, step):
    folder = Path(folder)
    normalizer.save(folder / f'normalizer_{step:09d}.pkl')
    rms = normalizer.obs_rms
    data = dict(settings=SETTINGS, steps=step, count=float(rms.count),
        mean=rms.mean.tolist(), variance=rms.var.tolist())
    path = folder / f'normalizer_{step:09d}.json'
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


class NormalizedIDACReplay:
    """Preserve the upstream raw buffer and sampler RNG; normalize sampled inputs."""
    def __init__(self, replay, normalizer):
        self.raw = replay
        self.normalizer = normalizer

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def sample(self, batch_size):
        import torch
        r = self.raw
        indices = np.random.randint(0, r.size, size=batch_size)
        fields = (self.normalizer.normalize_obs(r.state[indices]), r.action[indices],
                  self.normalizer.normalize_obs(r.next_state[indices]),
                  r.reward[indices], r.not_done[indices])
        return tuple(torch.as_tensor(x, dtype=torch.float32, device=r.device) for x in fields)


def normalized_ibolt_class(base):
    class NormalizedFetchIBOLT(base):
        def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
            super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
            # VecNormalize clips terminal_observation. Recover the actual physical
            # next state, rather than inverse-transforming a clipped terminal state.
            assert not replay_buffer.optimize_memory_usage
            index = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            for i, info in enumerate(infos):
                replay_buffer.next_observations[index, i] = info['fetch_raw_next_observation']

        def _save_model(self):
            super()._save_model()
            normalizer = self.get_vec_normalize_env()
            if normalizer is not None:
                save_normalizer(normalizer, self.model_save_path, self.num_timesteps)
    return NormalizedFetchIBOLT
