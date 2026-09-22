"""Only bookkeeping wrappers; physics, map, reward and resets are upstream."""
from functools import partial
from pathlib import Path
import sys
import copy
import copyreg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "antmaze"))
import gym
import ddiffpg  # register exact upstream environments


def restore_gym_rng(state):
    """Gym0.23's pickle constructor predates NumPy1.24's second argument."""
    from gym.utils.seeding import RandomNumberGenerator
    bitgen = getattr(np.random, state['bit_generator'])()
    bitgen.state = state
    return RandomNumberGenerator(bitgen)


def reduce_gym_rng(rng):
    return restore_gym_rng, (rng.bit_generator.state,)


from gym.utils.seeding import RandomNumberGenerator
copyreg.pickle(RandomNumberGenerator, reduce_gym_rng)


class Recorded(gym.Wrapper):
    def __init__(self, env, fixed=False):
        super().__init__(env)
        self.fixed = fixed
        self.initial = None
        self.length = 0

    @property
    def physics_env(self):
        return self.env.unwrapped.wrapped_env

    def state(self):
        e = self.physics_env
        s = e.sim.get_state()
        return dict(time=s.time, qpos=s.qpos.copy(), qvel=s.qvel.copy(),
                    act=None if s.act is None else s.act.copy(),
                    udd_state=copy.deepcopy(s.udd_state),
                    rng=copy.deepcopy(e.np_random.bit_generator.state),
                    length=self.length, elapsed=self.env._elapsed_steps)

    def restore(self, state):
        import mujoco_py
        e = self.physics_env
        e.sim.set_state(mujoco_py.MjSimState(state['time'], state['qpos'],
            state['qvel'], state['act'], state['udd_state']))
        e.sim.forward()
        e.np_random.bit_generator.state = copy.deepcopy(state['rng'])
        self.length = state['length']
        self.env._elapsed_steps = state['elapsed']
        return e._get_obs().astype(np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.length = 0
        if self.fixed:
            if self.initial is None:
                self.initial = self.state()
            else:
                obs = self.restore(self.initial)
        return np.asarray(obs, np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.length += 1
        info = dict(info, xy=np.asarray(obs[:2], np.float32), episode_length=self.length)
        assert reward in (0, 10, 20), reward
        return np.asarray(obs, np.float32), reward, done, info


def make_one(task, seed, fixed=False):
    # Upstream preprocess_cfg explicitly enables random starts only on v1.
    env = gym.make('antmaze-' + task, reward_type='sparse', random_init=task == 'v1')
    env.seed(seed)
    env.action_space.seed(seed)
    return Recorded(env, fixed=fixed)


def vector(task, count, seed, asynchronous=True, fixed=False):
    constructors = [partial(make_one, task, seed if fixed else seed + i, fixed)
                    for i in range(count)]
    if asynchronous:
        # Construct before initializing CUDA; workers execute CPU MuJoCo only.
        return gym.vector.AsyncVectorEnv(constructors, context='fork')
    return gym.vector.SyncVectorEnv(constructors)


def transition(next_obs, dones, infos):
    final = np.array(next_obs, dtype=np.float32, copy=True)
    terminal = np.asarray(dones, bool).copy()
    for i, info in enumerate(infos):
        if dones[i]:
            # Gym auto-resets; replay/NovelD must see the actual final state.
            final[i] = info['terminal_observation']
        if info.get('TimeLimit.truncated', False):
            terminal[i] = False
    return final, terminal
