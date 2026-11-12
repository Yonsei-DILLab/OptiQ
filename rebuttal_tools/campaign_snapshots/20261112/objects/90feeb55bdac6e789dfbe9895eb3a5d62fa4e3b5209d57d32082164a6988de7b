"""Check the pinned native tasks, seeded resets, horizons and EGL rendering."""
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from .run import TASKS, atomic, make_env


def main():
    rows = []
    for task, name in TASKS.items():
        env, paired = make_env(task), make_env(task)
        try:
            a, _ = env.reset(seed=8123)
            b, _ = paired.reset(seed=8123)
            assert np.array_equal(a, b)
            assert np.isfinite(a).all()
            assert env.spec.max_episode_steps == 200
            assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
            env.action_space.seed(41)
            for _ in range(200):
                action = env.action_space.sample()
                x = env.step(action)
                y = paired.step(action)
                assert np.array_equal(x[0], y[0]) and x[1:4] == y[1:4]
                assert 'success' in x[4] and np.isfinite(x[0]).all() and np.isfinite(x[1])
                if x[2] or x[3]:
                    break
            assert x[2] or x[3]
            render = gym.make(name, reward_type='sparse', render_mode='rgb_array', width=160, height=120)
            try:
                render.reset(seed=8123)
                frame = render.render()
                assert frame.shape == (120, 160, 3) and frame.dtype == np.uint8
            finally:
                render.close()
            rows.append(dict(task=task, env=name, observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0], horizon=200,
                reset_and_trajectory_reproducible=True, native_success_available=True, egl_render=True))
        finally:
            env.close()
            paired.close()
    atomic(Path('/workspace/heechan-adroit/environment-checks.json'), dict(passed=True, tasks=rows))
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
