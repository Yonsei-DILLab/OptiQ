"""Goal-aware Fetch input; user-requested Slide success radius is 20 cm."""
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
    env = gym.make(TASKS[task], reward_type='sparse')
    if task == 'slide':
        # FetchSlide fixes this constructor argument internally. Both sparse
        # reward and success use this same attribute, including HER relabeling.
        env.unwrapped.distance_threshold = 0.20
    return CompactGoal(env)
