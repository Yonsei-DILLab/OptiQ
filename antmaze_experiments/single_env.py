"""One native simulator, no Gym VectorEnv and no subprocess workers.

Singleton batch axes are only an interface to the unchanged learner/replay.
"""
import numpy as np
from .envs import make_one


class SingleEnv:
    num_envs = 1

    def __init__(self, task, seed, reward_profile, random_init):
        self.env = make_one(task, seed, reward_profile=reward_profile,
                            random_init=random_init)
        self.single_observation_space = self.env.observation_space
        self.single_action_space = self.env.action_space

    def reset(self):
        return np.asarray(self.env.reset())[None]

    def step(self, actions):
        assert np.asarray(actions).shape == (1, 8)
        obs, reward, done, info = self.env.step(actions[0])
        info = dict(info)
        if done:
            info['terminal_observation'] = np.asarray(obs).copy()
            obs = self.env.reset()
        return (np.asarray(obs)[None], np.asarray([reward]),
                np.asarray([done]), [info])

    def call(self, name, *args, **kwargs):
        return (getattr(self.env, name)(*args, **kwargs),)

    def close(self):
        self.env.close()
