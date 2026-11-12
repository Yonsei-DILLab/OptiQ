"""Shared future HER for Fetch; native learner updates remain unchanged."""
import json
import numpy as np
from .baselines.replay import Replay

SETTINGS = dict(strategy='future', n_sampled_goal=4, relabel_fraction=0.8,
    future='inclusive: next achieved goal from this or a later transition in the same episode',
    sampling='completed episodes only', reward='original environment compute_reward',
    normalization='relabel raw goals first, then normalize both observations with current training statistics',
    termination='preserve environment terminal masks; reaching a relabeled goal does not terminate Fetch')


def reward_function(env):
    if hasattr(env, 'venv'):
        env = env.venv.envs[0]
    return env.unwrapped.compute_reward


class FutureGoals:
    def __init__(self, capacity, task, seed, compute_reward):
        self.capacity = capacity
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.compute_reward = compute_reward
        self.total = self.completed_until = self.episode_start = 0
        self.ends = np.full(capacity, -1, np.int64)
        self.goal_slice = slice(0, 3) if task == 'reach' else slice(3, 6)
        self.last = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state['compute_reward'] = None
        return state

    def record(self, episode_end):
        index = self.total % self.capacity
        self.ends[index] = -1
        self.total += 1
        if episode_end:
            start = max(self.episode_start, self.total-self.capacity)
            self.ends[np.arange(start, self.total) % self.capacity] = self.total
            self.completed_until = self.total
            self.episode_start = self.total

    def draw(self, batch_size):
        oldest = max(0, self.total-self.capacity)
        if self.completed_until <= oldest:
            raise RuntimeError('HER requires a completed episode before sampling')
        counters = self.rng.integers(oldest, self.completed_until, size=batch_size)
        indices = counters % self.capacity
        n_her = int(batch_size*SETTINGS['relabel_fraction'])
        ends = self.ends[indices[:n_her]]
        assert np.all(ends > counters[:n_her])
        future = counters[:n_her] + (self.rng.random(n_her)*(ends-counters[:n_her])).astype(np.int64)
        return indices, future % self.capacity, counters, future

    def relabel(self, data, raw_next, future_indices, counters, future):
        n_her = len(future_indices)
        goals = raw_next[future_indices, self.goal_slice].copy()
        data['observations'][:n_her, -3:] = goals
        data['next_observations'][:n_her, -3:] = goals
        if self.compute_reward is None:
            from .env import make_env
            env = make_env(self.task)
            self.compute_reward = env.unwrapped.compute_reward
            env.close()
        achieved = data['next_observations'][:n_her, self.goal_slice]
        data['rewards'][:n_her] = self.compute_reward(achieved, goals, [{} for _ in goals])
        self.last = dict(batch_size=len(counters), relabeled_samples=n_her,
            relabeled_zero_reward_fraction=float(np.mean(data['rewards'][:n_her] == 0)),
            completed_transitions=self.completed_until,
            sampled_future_within_episode=bool(np.all((future >= counters[:n_her]) &
                (future < self.ends[counters[:n_her] % self.capacity]))))
        return data

    def metadata(self):
        return dict(her_total=np.int64(self.total), her_completed_until=np.int64(self.completed_until),
            her_episode_start=np.int64(self.episode_start), her_episode_ends=self.ends,
            her_task=np.asarray(self.task), her_rng=np.asarray(json.dumps(self.rng.bit_generator.state)))


class HERReplay(Replay):
    def __init__(self, *args, task, compute_reward, **kwargs):
        super().__init__(*args, **kwargs)
        self.her = FutureGoals(self.capacity, task, kwargs.get('seed', 0), compute_reward)

    def add(self, observations, actions, rewards, next_observations, terminals, episode_ends=None):
        if len(observations) != 1 or episode_ends is None:
            raise ValueError('Fetch HER requires one environment and explicit episode boundaries')
        super().add(observations, actions, rewards, next_observations, terminals)
        self.her.record(bool(episode_ends[0]))
        assert self.position == self.her.total % self.capacity

    def batch(self, size):
        indices, future_indices, counters, future = self.her.draw(size)
        data = {key: value[indices].copy() for key, value in self.data.items()}
        data = self.her.relabel(data, self.data['next_observations'], future_indices, counters, future)
        if self.normalizer is not None:
            for key in ('observations', 'next_observations'):
                data[key] = self.normalizer.normalize_obs(data[key])
        return data

    def save(self, path):
        np.savez_compressed(path, **{k:v[:self.size] for k,v in self.data.items()},
            size=self.size, position=self.position, **self.her.metadata())


class IDACHERReplay:
    def __init__(self, replay, device):
        self.raw = replay
        self.device = device

    def sample(self, batch_size):
        import torch
        b = self.raw.batch(batch_size)
        fields = (b['observations'], b['actions'], b['next_observations'],
                  b['rewards'][:,None], 1.-b['terminals'][:,None])
        return tuple(torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in fields)


from stable_baselines3.common.buffers import ReplayBuffer


class SB3HERReplayBuffer(ReplayBuffer):
    def __init__(self, *args, task, compute_reward, her_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or self.optimize_memory_usage:
            raise ValueError('Fetch HER requires n_envs=1 and explicit next observations')
        self.her = FutureGoals(self.buffer_size, task, her_seed, compute_reward)

    def add(self, obs, next_obs, action, reward, done, infos):
        # Keep the true physical terminal state even when VecNormalize clips it.
        next_obs = np.asarray([info.get('fetch_raw_next_observation', next_obs[i])
                               for i, info in enumerate(infos)], np.float32)
        super().add(obs, next_obs, action, reward, done, infos)
        self.her.record(bool(done[0]))
        assert self.pos == self.her.total % self.buffer_size

    def sample(self, batch_size, env=None):
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        indices, future_indices, counters, future = self.her.draw(batch_size)
        data = dict(observations=self.observations[indices,0].copy(),
            actions=self.actions[indices,0].copy(),
            next_observations=self.next_observations[indices,0].copy(),
            rewards=self.rewards[indices,0].copy(),
            terminals=(self.dones[indices,0]*(1-self.timeouts[indices,0])).copy())
        data = self.her.relabel(data, self.next_observations[:,0], future_indices, counters, future)
        if env is not None:
            for key in ('observations','next_observations'):
                data[key] = env.normalize_obs(data[key])
            data['rewards'] = env.normalize_reward(data['rewards'])
        tensor = lambda k, column=False:self.to_torch(data[k][:,None] if column else data[k])
        return ReplayBufferSamples(tensor('observations'), tensor('actions'),
            tensor('next_observations'), tensor('terminals',True), tensor('rewards',True))
