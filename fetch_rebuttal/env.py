"""Goal-aware state input, unmodified sparse reward and time-limit semantics."""
import gymnasium as gym
import gymnasium_robotics
import numpy as np

TASKS = {'reach': 'FetchReach-v4', 'push': 'FetchPush-v4',
         'slide': 'FetchSlide-v4', 'pick-and-place': 'FetchPickAndPlace-v4'}

class CompactGoal(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        dim = env.observation_space['observation'].shape[0] + 3
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)

    def observation(self, obs):
        return np.concatenate((obs['observation'], obs['desired_goal'])).astype(np.float32)

def make_env(task):
    gym.register_envs(gymnasium_robotics)
    return CompactGoal(gym.make(TASKS[task], reward_type='sparse'))
