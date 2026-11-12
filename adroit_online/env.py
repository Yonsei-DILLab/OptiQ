"""Native environment construction; no demonstration or pretrained-policy imports."""
from .config import TASKS


def make_env(task):
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make(TASKS[task], reward_type='sparse', max_episode_steps=200)
    assert env.spec.id == TASKS[task] and env.spec.max_episode_steps == 200
    assert env.unwrapped.sparse_reward
    return env
