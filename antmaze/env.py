"""One unmodified official task and observation contract for all algorithms."""
import gymnasium as gym
import gymnasium_robotics
import numpy as np

ENV_ID = "AntMaze_UMaze-v5"
ENV_KWARGS = dict(reward_type="sparse", continuing_task=False, max_episode_steps=1000)
OBS_KEYS = ("observation", "achieved_goal", "desired_goal")


class GoalObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = [env.observation_space[k] for k in OBS_KEYS]
        self.observation_space = gym.spaces.Box(
            np.concatenate([s.low for s in spaces]).astype(np.float32),
            np.concatenate([s.high for s in spaces]).astype(np.float32), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs[k] for k in OBS_KEYS]).astype(np.float32)


def make_env(seed=None):
    gym.register_envs(gymnasium_robotics)
    env = GoalObservation(gym.make(ENV_ID, **ENV_KWARGS))
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env
