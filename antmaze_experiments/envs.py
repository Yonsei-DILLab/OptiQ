"""Original physics/resets with explicit sparse or dense-distance reward."""
from functools import partial
from pathlib import Path
import sys
import copy
import copyreg
import numpy as np
from .progress_reward import PROFILES, progress_reward, geodesic, maze_geometry, bonus_enabled, is_geodesic

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
    def __init__(self, env, fixed=False, reward_profile='sparse', task=None):
        super().__init__(env)
        self.fixed = fixed
        self.initial = None
        self.length = 0
        self.reward_profile = reward_profile
        if reward_profile in PROFILES:
            self.task = task
            assert task in ("v1","v2","v3","v4")
            assert self.physics_env._maze_size_scaling==4
            actual_goals=np.asarray(self.physics_env.target_goal).reshape(-1,2)
            np.testing.assert_array_equal(actual_goals,maze_geometry(task)[1])

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
        before = self.physics_env.get_xy().copy() if self.reward_profile in PROFILES else None
        obs, reward, done, info = self.env.step(action)
        self.length += 1
        assert reward in (0, 10, 20), reward
        goals = np.asarray(self.physics_env.target_goal).reshape(-1,2)
        distance = float(np.linalg.norm(np.asarray(obs[:2])-goals,axis=1).min())
        info = dict(info, xy=np.asarray(obs[:2], np.float32), episode_length=self.length,
                    distance=distance,upstream_sparse_reward=float(reward))
        if self.reward_profile == 'dense':
            reward = -distance
        elif self.reward_profile in PROFILES:
            reward, previous_distance, current_distance = progress_reward(
                before, self.physics_env.get_xy(), self.task, self.reward_profile, reward)
            info.update(reward_progress=float(previous_distance-current_distance),
                reward_step_penalty=-.01,reward_success=info['upstream_sparse_reward'] if bonus_enabled(self.reward_profile) else 0.,
                progress_distance_before=float(previous_distance),
                progress_distance_after=float(current_distance),reward_profile=self.reward_profile)
            if info['success']:
                assert done, 'Success bonus must terminate, not repeat each step'
        assert np.isfinite(reward)
        return np.asarray(obs, np.float32), reward, done, info


def make_one(task, seed, fixed=False, reward_profile='sparse', random_init=None):
    # Upstream preprocess_cfg explicitly enables random starts only on v1.
    reset_random = task == 'v1' if random_init is None else bool(random_init)
    env = gym.make('antmaze-' + task, reward_type='sparse', random_init=reset_random)
    env.seed(seed)
    # Gym0.23 Env.seed is a no-op beneath D4RL's ProxyEnv. Seed the actual
    # MuJoCo environment RNG used by the original reset_model implementation.
    env.unwrapped.wrapped_env.np_random, _ = gym.utils.seeding.np_random(seed)
    env.action_space.seed(seed)
    assert reward_profile in ('sparse','dense') + PROFILES
    return Recorded(env, fixed=fixed, reward_profile=reward_profile, task=task)


def vector(task, count, seed, asynchronous=True, fixed=False, reward_profile='sparse',random_init=None):
    if is_geodesic(reward_profile):
        geodesic(task)  # Precompute once before fork; workers inherit immutable graph.
    constructors = [partial(make_one, task, seed if fixed else seed + i, fixed, reward_profile,random_init)
                    for i in range(count)]
    if asynchronous:
        # Construct before initializing CUDA; workers execute CPU MuJoCo only.
        import resource
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        required=max(4096,count*8+512)
        if soft<required:
            if hard!=resource.RLIM_INFINITY and hard<required:
                raise RuntimeError(f'256-env execution requires file limit>={required}; hard limit={hard}')
            resource.setrlimit(resource.RLIMIT_NOFILE,(required,hard))
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
