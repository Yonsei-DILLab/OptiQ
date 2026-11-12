"""Immutable demonstration replay with exactly half of every optimizer minibatch."""
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class DemonstrationReplay(ReplayBuffer):
    def __init__(self, *args, demo_path, demo_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        with np.load(demo_path, allow_pickle=False) as d:
            self.demo = {k: d[k].copy() for k in ('observations', 'actions', 'next_observations', 'dones', 'rewards')}
        assert all(np.isfinite(x).all() for x in self.demo.values())
        assert self.demo['observations'].shape[1:] == self.obs_shape
        assert self.demo['actions'].shape[1] == self.action_dim
        assert len(self.demo['actions']) > 0
        self.demo_rng = np.random.default_rng(demo_seed)
        self.last_mix = None
        self.online_transitions = self.online_success_steps = self.online_episodes = self.online_success_episodes = 0
        self.first_online_success_at = None
        self.episode_success = np.zeros(self.n_envs, bool)

    def add(self, obs, next_obs, action, reward, done, infos):
        success = np.asarray([i['success'] for i in infos], bool)
        self.online_transitions += self.n_envs
        self.online_success_steps += int(success.sum())
        self.episode_success |= success
        if self.first_online_success_at is None and success.any():
            self.first_online_success_at = self.online_transitions
        self.online_episodes += int(np.asarray(done).sum())
        self.online_success_episodes += int((self.episode_success & done).sum())
        self.episode_success[done] = False
        return super().add(obs, next_obs, action, reward, done, infos)

    def discovery(self):
        return dict(online_transitions=self.online_transitions, online_success_steps=self.online_success_steps,
            online_episodes=self.online_episodes, online_success_episodes=self.online_success_episodes,
            first_online_success_at_transitions=self.first_online_success_at)

    def sample(self, batch_size, env=None):
        assert batch_size % 256 == 0, 'Every optimizer minibatch must have 256 samples'
        assert env is None, 'The prepared profile does not normalize observations or rewards'
        online = super().sample(batch_size//2, env=env)
        indices = self.demo_rng.integers(0, len(self.demo['actions']), batch_size//2)
        rows = []
        for key, values in zip(('observations', 'actions', 'next_observations', 'dones', 'rewards'), online):
            offline = self.to_torch(self.demo[key][indices]).to(dtype=values.dtype)
            if key in ('dones', 'rewards'):
                offline = offline.reshape(-1, 1)
            mixed = torch.empty((batch_size, *values.shape[1:]), dtype=values.dtype, device=values.device)
            mixed[0::2], mixed[1::2] = offline, values
            rows.append(mixed)
        self.last_mix = dict(batch_size=batch_size, demonstration_samples=batch_size//2,
                             online_samples=batch_size//2, per_optimizer_minibatch=[128, 128])
        return ReplayBufferSamples(*rows)
