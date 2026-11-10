"""Keep the existing HER sampler; choose achieved-goal coordinates by task."""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from rebuttal_tools.fetch_vector_replay import VectorHERReplayBuffer as PinnedReplay


class VectorHERReplayBuffer(PinnedReplay):
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self.achieved = slice(0, 3) if task == 'reach' else slice(3, 6)

    def add(self, obs, next_obs, action, reward, done, infos):
        if self.task != 'reach':
            return super().add(obs, next_obs, action, reward, done, infos)
        # Reach has no object; keep episode metadata without object-lift metrics.
        next_obs = np.asarray([i['fetch_raw_next_observation'] for i in infos], np.float32)
        self.episode_ends[self.pos] = -1
        ReplayBuffer.add(self, obs, next_obs, action, reward, done, infos)
        self.total_vector_steps += 1
        self.training_episodes += int(np.sum(done))
        for e in np.flatnonzero(done):
            start = max(int(self.episode_starts[e]), self.total_vector_steps-self.buffer_size)
            times = np.arange(start, self.total_vector_steps)
            self.episode_ends[times % self.buffer_size, e] = self.total_vector_steps
            self.episode_starts[e] = self.last_completed_ends[e] = self.total_vector_steps

    def sample(self, batch_size, env=None):
        times, envs, future, her = self.draw(batch_size)
        slots = times % self.buffer_size
        obs = self.observations[slots, envs].copy()
        nxt = self.next_observations[slots, envs].copy()
        actions = self.actions[slots, envs].copy()
        rewards = self.rewards[slots, envs].copy()
        dones = (self.dones[slots, envs]*(1-self.timeouts[slots, envs])).copy()
        goals = self.next_observations[future % self.buffer_size, envs[her], self.achieved]
        obs[her, -3:] = goals
        nxt[her, -3:] = goals
        if self.compute_reward is None:
            from fetch_rebuttal.env import make_env
            reward_env = make_env(self.task)
            self.compute_reward = reward_env.unwrapped.compute_reward
            reward_env.close()
        rewards[her] = self.compute_reward(nxt[her, self.achieved], goals, [{} for _ in her])
        self.last = dict(batch_size=batch_size, relabeled_samples=len(her),
            relabeled_per_optimizer_minibatch=204,
            relabeled_zero_reward_fraction=float(np.mean(rewards[her] == 0)),
            sampled_future_within_episode=True, sampled_future_same_environment=True,
            task=self.task, achieved_goal_slice=[self.achieved.start, self.achieved.stop],
            total_transitions=self.total_vector_steps*self.n_envs)
        if env is not None:
            obs = env.normalize_obs(obs)
            nxt = env.normalize_obs(nxt)
            rewards = env.normalize_reward(rewards)
        tensor = self.to_torch
        return ReplayBufferSamples(tensor(obs), tensor(actions), tensor(nxt),
            tensor(dones[:, None]), tensor(rewards[:, None]))


class NativeReplayView:
    """Present normalized shared HER minibatches to unchanged MFPO/MEOW updates."""
    def __init__(self, replay, normalizer):
        self.replay, self.normalizer = replay, normalizer

    def batch(self, batch_size):
        sample = self.replay.sample(batch_size, env=self.normalizer)
        return dict(observations=sample.observations.numpy(),
            actions=sample.actions.numpy(), next_observations=sample.next_observations.numpy(),
            rewards=sample.rewards.numpy().reshape(-1), terminals=sample.dones.numpy().reshape(-1))
