"""Normalize Fetch from completed, HER-relabeled episodes without changing learners.

The OpenAI HER recipe samples completed episodes to update both observation and
goal statistics. Its standard-deviation floor is 0.01, with raw clipping at 200
and normalized clipping at 5. The training replay and learner RNGs remain separate.
"""
from pathlib import Path
import json
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize


class HERGoalNormalize(VecNormalize):
    def __init__(self, venv, *, task, stats_seed, replay_k=4,
                 min_std=0.01, raw_clip=200.0, **kwargs):
        kwargs.setdefault('norm_reward', False)
        assert kwargs.get('norm_obs', True)
        assert not kwargs['norm_reward']
        super().__init__(venv, **kwargs)
        assert not isinstance(self.obs_rms, dict)
        self.task = task
        self.achieved_slice = slice(0, 3) if task == 'reach' else slice(3, 6)
        self.stats_rng = np.random.default_rng(np.random.SeedSequence([stats_seed, 1162026]))
        self.stats_replay_k = int(replay_k)
        self.min_std = float(min_std)
        self.raw_clip = float(raw_clip)
        self.completed_stats_episodes = 0
        self.stats_transitions = 0
        self.last_stats_audit = {}
        self._episode_observations = None
        self._episode_achieved = None
        self._previous_raw = None

    def _normalization_input(self, obs):
        return np.asarray(obs)

    def normalize_obs(self, obs):
        if not self.norm_obs:
            return np.array(obs, copy=True)
        raw = np.clip(self._normalization_input(obs), -self.raw_clip, self.raw_clip)
        std = np.maximum(np.sqrt(np.maximum(self.obs_rms.var, 0.0)), self.min_std)
        return np.clip((raw - self.obs_rms.mean) / std,
                       -self.clip_obs, self.clip_obs).astype(np.float32)

    def unnormalize_obs(self, obs):
        if not self.norm_obs:
            return np.array(obs, copy=True)
        std = np.maximum(np.sqrt(np.maximum(self.obs_rms.var, 0.0)), self.min_std)
        return np.asarray(obs) * std + self.obs_rms.mean

    def reset(self):
        # VecNormalize's live-observation update would count desired goals only.
        # Use completed HER episodes as the sole statistics source instead.
        training = self.training
        self.training = False
        try:
            super().reset()
        finally:
            self.training = training
        raw = self.get_original_obs().copy()
        self._previous_raw = raw
        self._episode_observations = [[] for _ in range(self.num_envs)]
        self._episode_achieved = [[row[self.achieved_slice].copy()] for row in raw]
        return self.normalize_obs(raw)

    def _record_completed_episodes(self, raw, dones, infos):
        completed = []
        for i in range(self.num_envs):
            nxt = np.asarray(infos[i]['fetch_raw_next_observation'])
            self._episode_observations[i].append(self._previous_raw[i].copy())
            self._episode_achieved[i].append(nxt[self.achieved_slice].copy())
            if dones[i]:
                completed.append((np.asarray(self._episode_observations[i]),
                                  np.asarray(self._episode_achieved[i])))
                self._episode_observations[i] = []
                self._episode_achieved[i] = [raw[i, self.achieved_slice].copy()]
        self._previous_raw = raw.copy()
        if not completed:
            return
        lengths = {len(o) for o, _ in completed}
        assert len(lengths) == 1, 'Fetch uses one fixed episode horizon'
        horizon = lengths.pop()
        observations = np.stack([o for o, _ in completed])
        achieved = np.stack([ag for _, ag in completed])
        count = len(completed) * horizon
        episode = self.stats_rng.integers(0, len(completed), count)
        times = self.stats_rng.integers(0, horizon, count)
        her = self.stats_rng.random(count) < (1.0 - 1.0 / (1.0 + self.stats_replay_k))
        future = times + 1 + (self.stats_rng.random(count) * (horizon - times)).astype(np.int64)
        sampled = observations[episode, times].copy()
        original_goals = sampled[:, -3:].copy()
        sampled[her, -3:] = achieved[episode[her], future[her]]
        inputs = self._normalization_input(sampled)
        self.obs_rms.update(np.clip(inputs, -self.raw_clip, self.raw_clip))
        self.completed_stats_episodes += len(completed)
        self.stats_transitions += count
        self.last_stats_audit = dict(
            completed_episodes=len(completed), horizon=horizon,
            sampled_transitions=count, relabeled_goals=int(her.sum()),
            future_same_episode=True, terminal_achieved_goal_included=True,
            original_goal_mean=original_goals.mean(0).tolist(),
            sampled_goal_mean=sampled[:, -3:].mean(0).tolist(),
            effective_std_min=float(np.maximum(np.sqrt(self.obs_rms.var), self.min_std).min()))

    def step_wait(self):
        training = self.training
        self.training = False
        try:
            _, rewards, dones, infos = super().step_wait()
        finally:
            self.training = training
        raw = self.get_original_obs().copy()
        if training:
            self._record_completed_episodes(raw, dones, infos)
        else:
            # Evaluation never updates moments; discard its partial episodes.
            self._previous_raw = raw
            self._episode_observations = [[] for _ in range(self.num_envs)]
            self._episode_achieved = [[row[self.achieved_slice].copy()] for row in raw]
        # Apply the newly updated statistics consistently to autoreset and
        # physical terminal observations; never invert an already clipped value.
        for i, done in enumerate(dones):
            if done:
                infos[i]['terminal_observation'] = self.normalize_obs(
                    infos[i]['fetch_raw_next_observation'])
        return self.normalize_obs(raw), rewards, dones, infos


class HERRelativeGoalNormalize(HERGoalNormalize):
    """Express the goal relative to the achieved position after raw HER relabeling."""

    def _normalization_input(self, obs):
        inputs = np.array(obs, copy=True)
        inputs[..., -3:] -= inputs[..., self.achieved_slice]
        return inputs

    def unnormalize_obs(self, obs):
        raw = super().unnormalize_obs(obs)
        if self.norm_obs:
            raw[..., -3:] += raw[..., self.achieved_slice]
        return raw


def save_normalizer(normalizer, folder, step):
    folder = Path(folder)
    normalizer.save(folder / f'normalizer_{step:09d}.pkl')
    rms = normalizer.obs_rms
    data = dict(steps=step, count=float(rms.count), mean=rms.mean.tolist(),
                variance=rms.var.tolist(),
                settings=dict(implementation=type(normalizer).__name__, norm_obs=True,
                    norm_reward=False, clip_obs=normalizer.clip_obs,
                    raw_clip_obs=normalizer.raw_clip, min_std=normalizer.min_std,
                    statistics='Completed-episode observations and future-HER goals; frozen during evaluation',
                    replay_k=normalizer.stats_replay_k,
                    formula='clip((clip(raw,-200,200)-mean)/max(sqrt(var),0.01),-5,5)'),
                stats_transitions=normalizer.stats_transitions,
                completed_stats_episodes=normalizer.completed_stats_episodes,
                last_stats_audit=normalizer.last_stats_audit)
    if isinstance(normalizer, HERRelativeGoalNormalize):
        data['settings']['goal_coordinates'] = 'desired goal minus achieved goal in this observation'
    (folder / f'normalizer_{step:09d}.json').write_text(json.dumps(data, indent=2) + '\n')


def normalized_ibolt_class(base):
    from fetch_rebuttal.normalization import normalized_ibolt_class as raw_transition_class
    parent = raw_transition_class(base)

    class HERNormalizedFetchIBOLT(parent):
        def _save_model(self):
            base._save_model(self)
            normalizer = self.get_vec_normalize_env()
            if normalizer is not None:
                save_normalizer(normalizer, self.model_save_path, self.num_timesteps)

    return HERNormalizedFetchIBOLT
