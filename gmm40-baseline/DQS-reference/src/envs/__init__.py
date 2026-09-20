# Code adapted from https://github.com/ikostrikov/jaxrl

from typing import Optional

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RescaleAction

from src.envs import wrappers
from src.envs.gmm_env import GMMEnv
from src.envs.dmc_env import DMCEnv


gym.register("GMM-v0", entry_point=GMMEnv)


POINTMAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'g', 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 'r', 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 1, 'g', 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def make_gmm_env(env_id):
    def thunk(seed):
        env = gym.make(env_id)
        return env

    return thunk


def make_pointmaze_env():
    def thunk(seed):
        env = gym.make('PointMaze_Medium_Diverse_G-v3', maze_map=POINTMAZE_MAP, reward_type='sparse')
        return env
    return thunk


def make_dmc_env(env_name: str,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             flatten: bool = True) -> gym.Env:
    def thunk(seed):
        all_envs = gym.envs.registry.values()
        env_ids = [env_spec.id for env_spec in all_envs]

        if env_name in env_ids:
            env = gym.make(env_name, render_mode="rgb_array")
        else:
            domain_name, task_name = env_name.split('-')
            env = DMCEnv(domain=domain_name,
                                task=task_name,
                                task_kwargs={'random': seed})
        if flatten and isinstance(env.observation_space, gym.spaces.Dict):
            env = gym.wrappers.FlattenObservation(env)

        if add_episode_monitor:
            env = wrappers.EpisodeMonitor(env)

        if action_repeat > 1:
            env = wrappers.RepeatAction(env, action_repeat)

        env = RescaleAction(env, -1.0, 1.0)

        if save_folder is not None:
            env = gym.wrappers.RecordVideo(env, save_folder)
        env = wrappers.SinglePrecision(env)

        try:
            env.reset(seed=seed)
        except TypeError:
            # Some envs may ignore seed kwarg; that's fine
            env.reset()
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk