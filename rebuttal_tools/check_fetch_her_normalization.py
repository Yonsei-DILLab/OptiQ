"""Independent checks for completed-episode HER normalization; no training."""
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from rebuttal_tools.fetch_her_normalization import HERGoalNormalize, HERRelativeGoalNormalize, save_normalizer


class EpisodeProbe(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (28,), np.float32)
    action_space = spaces.Box(-1, 1, (4,), np.float32)

    def __init__(self, identifier):
        self.identifier = identifier
        self.episode = -1

    def observation(self):
        row = np.zeros(28, np.float32)
        row[0] = self.identifier
        row[1] = self.time
        row[3:6] = [self.identifier * 10 + self.episode, self.time * .01, .4]
        row[-3:] = [100 + self.identifier, 10, .4]
        return row

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.time = 0
        return self.observation(), {}

    def step(self, action):
        self.time += 1
        obs = self.observation()
        return obs, -1., False, self.time == 50, {'fetch_raw_next_observation': obs.copy()}


def check_relative_goals():
    def observations_seen_by_normalizer(normalizer_class):
        env = normalizer_class(DummyVecEnv([lambda: EpisodeProbe(0)]),
                               task='slide', stats_seed=5)
        try:
            samples = []
            env.obs_rms.update = lambda values: samples.append(values.copy())
            env.reset()
            for _ in range(50):
                _, _, _, infos = env.step(np.zeros((1, 4), np.float32))
            assert len(samples) == 1
            np.testing.assert_array_equal(
                infos[0]['fetch_raw_next_observation'][-3:], np.array([100, 10, .4], np.float32))
            if isinstance(env, HERRelativeGoalNormalize):
                raw = np.zeros((2, 28), np.float32)
                raw[:, 3:6] = [[.5, .25, .4], [1., .25, .4]]
                raw[:, -3:] = [1.25, .75, .4]
                original = raw.copy()
                normalized = env.normalize_obs(raw)
                np.testing.assert_array_equal(normalized[:, -3:], [[.75, .5, 0.], [.25, .5, 0.]])
                np.testing.assert_array_equal(raw, original)
                np.testing.assert_allclose(env.unnormalize_obs(normalized), original)
                del env.obs_rms.update
                with tempfile.TemporaryDirectory() as directory:
                    save_normalizer(env, directory, 50)
                    restored = HERGoalNormalize.load(Path(directory) / 'normalizer_000000050.pkl',
                                                       DummyVecEnv([lambda: EpisodeProbe(0)]))
                    try:
                        assert isinstance(restored, HERRelativeGoalNormalize)
                        np.testing.assert_array_equal(restored.normalize_obs(raw), normalized)
                    finally:
                        restored.close()
            return samples[0]
        finally:
            env.close()

    absolute = observations_seen_by_normalizer(HERGoalNormalize)
    relative = observations_seen_by_normalizer(HERRelativeGoalNormalize)
    expected = absolute.copy()
    expected[:, -3:] -= absolute[:, 3:6]
    np.testing.assert_array_equal(relative, expected)


def run():
    defaults = HERGoalNormalize(DummyVecEnv([lambda: EpisodeProbe(0)]),
                                task='slide', stats_seed=5)
    try:
        assert defaults.norm_obs and not defaults.norm_reward
    finally:
        defaults.close()
    check_relative_goals()
    env = HERGoalNormalize(DummyVecEnv([lambda i=i: EpisodeProbe(i) for i in range(4)]),
                           task='slide', stats_seed=5, norm_obs=True, norm_reward=False,
                           clip_obs=5., epsilon=1e-8)
    try:
        env.reset()
        count = env.obs_rms.count
        captured = []
        update = env.obs_rms.update

        def capture(values):
            captured.append(values.copy())
            update(values)

        env.obs_rms.update = capture
        np_state = copy.deepcopy(np.random.get_state())
        for _ in range(49):
            env.step(np.zeros((4, 4), np.float32))
        assert env.obs_rms.count == count, 'Incomplete episodes entered statistics'
        obs, rewards, dones, infos = env.step(np.zeros((4, 4), np.float32))
        assert dones.all() and env.completed_stats_episodes == 4
        assert env.stats_transitions == 200 and env.obs_rms.count == count + 200
        sample = captured[-1]
        her = sample[:, -3] < 100
        assert 120 < her.sum() < 190, 'Expected an 80% HER goal mixture'
        np.testing.assert_array_equal(sample[her, -3], sample[her, 0] * 10)
        assert np.all(sample[her, -2] >= (sample[her, 1] + 1) * .01 - 1e-7)
        assert np.all(sample[her, -2] <= .5 + 1e-7)
        np.testing.assert_array_equal(sample[~her, -3], 100 + sample[~her, 0])
        assert sample[:, -3].mean() < 50, 'Desired goals alone were used for statistics'
        for before, after in zip(np_state, np.random.get_state()):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(rewards, np.full(4, -1.))
        for i in range(4):
            np.testing.assert_array_equal(infos[i]['terminal_observation'],
                                          env.normalize_obs(infos[i]['fetch_raw_next_observation']))
        state = (env.obs_rms.mean.copy(), env.obs_rms.var.copy(), env.obs_rms.count)
        for _ in range(10):
            env.normalize_obs(np.ones((4, 28), np.float32))
        env.training = False
        env.step(np.zeros((4, 4), np.float32))
        for before, after in zip(state, (env.obs_rms.mean, env.obs_rms.var, env.obs_rms.count)):
            np.testing.assert_array_equal(before, after)
        del env.obs_rms.update  # remove the test-only local capture function before serialization
        with tempfile.TemporaryDirectory() as directory:
            save_normalizer(env, directory, 200)
            data = json.loads((Path(directory) / 'normalizer_000000200.json').read_text())
            assert data['settings']['min_std'] == .01
            restored = HERGoalNormalize.load(Path(directory) / 'normalizer_000000200.pkl',
                DummyVecEnv([lambda i=i: EpisodeProbe(i) for i in range(4)]))
            try:
                np.testing.assert_array_equal(restored.normalize_obs(np.ones((4, 28))),
                                              env.normalize_obs(np.ones((4, 28))))
                assert restored.stats_transitions == 200
            finally:
                restored.close()
        env.obs_rms.mean = np.zeros(28)
        env.obs_rms.var = np.full(28, 1e-10)
        np.testing.assert_allclose(env.normalize_obs(np.full((1, 28), .001)), .1, atol=1e-7)
        np.testing.assert_allclose(env.unnormalize_obs(env.normalize_obs(np.full((1, 28), .001))),
                                   .001, atol=1e-9)
        env.obs_rms.var = np.full(28, 10000.)
        np.testing.assert_allclose(env.normalize_obs(np.full((1, 28), 1000.)), 2., atol=1e-7)
        return dict(passed=True, optimizer_updates=0, checks=[
            'Statistics use completed episodes and both actual and HER goals',
            'HER normalization goals stay in the same episode and environment',
            'Future goals include the physical terminal achieved position',
            'Global learner/action RNG is unchanged',
            'Raw reward and time-limit terminal observations are preserved',
            'Evaluation and sampling do not update normalization moments',
            'Full normalizer checkpoint reload is identical',
            'Relative inputs use the current achieved goal after HER relabeling and preserve raw coordinates',
            'Standard deviation floor is 0.01; raw clipping is 200'])
    finally:
        env.close()


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
