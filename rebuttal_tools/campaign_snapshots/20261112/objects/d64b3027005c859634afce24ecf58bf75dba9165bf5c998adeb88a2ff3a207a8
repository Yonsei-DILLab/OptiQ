"""Future HER with separate episode boundaries for every vector environment."""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class VectorHERReplayBuffer(ReplayBuffer):
    def __init__(self, *args, compute_reward, her_seed=0, table_height=.4247, minibatch_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.optimize_memory_usage
        self.compute_reward = compute_reward
        self.her_rng = np.random.default_rng(her_seed)
        self.total_vector_steps = 0
        self.episode_starts = np.zeros(self.n_envs, np.int64)
        self.last_completed_ends = np.zeros(self.n_envs, np.int64)
        self.episode_ends = np.full((self.buffer_size, self.n_envs), -1, np.int64)
        self.last = {}
        self.minibatch_size = minibatch_size
        self.table_height = table_height
        self.new_episode = np.ones(self.n_envs,bool)
        self.initial_object_z = np.zeros(self.n_envs)
        self.max_episode_lift = np.zeros(self.n_envs)
        self.training_episodes = self.lifted_episodes = self.training_success_steps = 0
        self.first_lift_at_transitions = None

    def __getstate__(self):
        state = dict(self.__dict__);state['compute_reward'] = None
        return state

    def add(self, obs, next_obs, action, reward, done, infos):
        next_obs = np.asarray([i.get('fetch_raw_next_observation', next_obs[k]) for k, i in enumerate(infos)], np.float32)
        self.initial_object_z = np.where(self.new_episode,obs[:,5],self.initial_object_z)
        self.max_episode_lift = np.maximum(np.where(self.new_episode,0.,self.max_episode_lift),next_obs[:,5]-self.initial_object_z)
        self.training_success_steps += int(np.sum(reward==0))
        if self.first_lift_at_transitions is None and np.any(self.max_episode_lift>.05):
            self.first_lift_at_transitions = (self.total_vector_steps+1)*self.n_envs
        self.training_episodes += int(np.sum(done))
        self.lifted_episodes += int(np.sum(done & (self.max_episode_lift>.05)))
        self.new_episode = np.asarray(done,bool).copy()
        self.episode_ends[self.pos] = -1
        super().add(obs, next_obs, action, reward, done, infos)
        self.total_vector_steps += 1
        for e in np.flatnonzero(done):
            start = max(int(self.episode_starts[e]), self.total_vector_steps-self.buffer_size)
            times = np.arange(start, self.total_vector_steps)
            self.episode_ends[times % self.buffer_size, e] = self.total_vector_steps
            self.episode_starts[e] = self.total_vector_steps
            self.last_completed_ends[e] = self.total_vector_steps

    def draw(self, batch_size):
        oldest = max(0, self.total_vector_steps-self.buffer_size)
        if not np.any(self.last_completed_ends > oldest):
            raise RuntimeError('HER requires at least one completed episode')
        ts, es, accepted = [], [], 0
        while accepted < batch_size:
            count = max(32, batch_size-accepted)
            times = self.her_rng.integers(oldest, self.total_vector_steps, count)
            envs = self.her_rng.integers(0, self.n_envs, count)
            ends = self.episode_ends[times % self.buffer_size, envs]
            valid = (ends > times) & (ends <= self.total_vector_steps)
            ts.append(times[valid]);es.append(envs[valid]);accepted += int(valid.sum())
        times = np.concatenate(ts)[:batch_size];envs = np.concatenate(es)[:batch_size]
        # Native iBOLT draws batch_size*gradient_steps and slices it into optimizer
        # minibatches. Preserve the original 204 HER samples in EACH batch of 256.
        her_indices = np.concatenate([np.arange(start,start+int(min(self.minibatch_size,batch_size-start)*.8))
            for start in range(0,batch_size,self.minibatch_size)])
        ends = self.episode_ends[times[her_indices] % self.buffer_size, envs[her_indices]]
        future = times[her_indices] + (self.her_rng.random(len(her_indices))*(ends-times[her_indices])).astype(np.int64)
        assert np.all(future >= times[her_indices]) and np.all(future < ends)
        assert np.all(self.episode_ends[future % self.buffer_size, envs[her_indices]] == ends)
        return times, envs, future, her_indices

    def sample(self, batch_size, env=None):
        times, envs, future, her_indices = self.draw(batch_size)
        slots = times % self.buffer_size;n_her = len(future)
        obs = self.observations[slots, envs].copy()
        nxt = self.next_observations[slots, envs].copy()
        actions = self.actions[slots, envs].copy()
        rewards = self.rewards[slots, envs].copy()
        dones = (self.dones[slots, envs]*(1-self.timeouts[slots, envs])).copy()
        goals = self.next_observations[future % self.buffer_size, envs[her_indices], 3:6]
        obs[her_indices, -3:] = goals;nxt[her_indices, -3:] = goals
        if self.compute_reward is None:
            from fetch_rebuttal.env import make_env
            reward_env = make_env('pick-and-place');self.compute_reward = reward_env.unwrapped.compute_reward
            reward_env.close()
        rewards[her_indices] = self.compute_reward(nxt[her_indices, 3:6], goals, [{} for _ in range(n_her)])
        self.last = dict(batch_size=batch_size, relabeled_samples=n_her,
            relabeled_zero_reward_fraction=float(np.mean(rewards[her_indices] == 0)),
            relabeled_per_optimizer_minibatch=int(self.minibatch_size*.8),
            sampled_future_within_episode=True, sampled_future_same_environment=True,
            total_transitions=self.total_vector_steps*self.n_envs,
            her_future_goals_above5cm=float(np.mean(goals[:,2]>self.table_height+.05)),
            training_episodes=self.training_episodes,lifted_episodes=self.lifted_episodes,
            first_lift_at_transitions=self.first_lift_at_transitions)
        if env is not None:
            obs = env.normalize_obs(obs);nxt = env.normalize_obs(nxt)
            rewards = env.normalize_reward(rewards)
        tensor = self.to_torch
        return ReplayBufferSamples(tensor(obs), tensor(actions), tensor(nxt), tensor(dones[:, None]), tensor(rewards[:, None]))
