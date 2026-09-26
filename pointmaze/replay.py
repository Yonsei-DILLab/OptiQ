"""Dimension-independent replay with batched writes and shared sample indices."""

from __future__ import annotations

import numpy as np


class Replay:
    def __init__(self, capacity=1_000_000, observation_dim=4, action_dim=2, seed=0,
                 device="cpu"):
        self.capacity, self.position, self.size = capacity, 0, 0
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.data = {
            "observations": np.empty((capacity, observation_dim), np.float32),
            "actions": np.empty((capacity, action_dim), np.float32),
            "rewards": np.empty((capacity,), np.float32),
            "next_observations": np.empty((capacity, observation_dim), np.float32),
            "terminals": np.empty((capacity,), np.float32),
        }

    def add(self, observations, actions, rewards, next_observations, terminals):
        values = dict(observations=observations, actions=actions, rewards=rewards,
                      next_observations=next_observations, terminals=terminals)
        count = len(observations)
        if not 0 < count <= self.capacity:
            raise ValueError(count)
        indices = (np.arange(count) + self.position) % self.capacity
        for key, value in values.items():
            value = np.asarray(value, np.float32)
            if value.shape != (count,) + self.data[key].shape[1:]:
                raise ValueError(f"{key}: {value.shape}")
            self.data[key][indices] = value
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def batch(self, size):
        if self.size < size:
            raise RuntimeError(f"replay has {self.size} rows, needs {size}")
        indices = self.rng.integers(self.size, size=size)
        return {key: value[indices].copy() for key, value in self.data.items()}

    def sample(self, batch_size, env=None):
        """Stable Baselines3 replay interface used by the iBOLT learner."""
        del env
        import torch
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        data = self.batch(batch_size)
        tensor = lambda key, column=False: torch.from_numpy(
            data[key][:, None] if column else data[key]).to(self.device)
        return ReplayBufferSamples(tensor("observations"), tensor("actions"),
                                   tensor("next_observations"), tensor("terminals", True),
                                   tensor("rewards", True))
