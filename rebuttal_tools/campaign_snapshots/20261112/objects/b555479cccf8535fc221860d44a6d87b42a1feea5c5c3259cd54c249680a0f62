"""Validate collector boundaries and demo mixing before launching the full cohort."""
from pathlib import Path
import numpy as np
from .demo_replay import DemonstrationReplay
from .run import atomic, make_env
from .vector_env import GroupedAdroitVecEnv


def main():
    a = GroupedAdroitVecEnv('door', 8, [[], []])
    b = GroupedAdroitVecEnv('door', 8, [[], [], [], []])
    try:
        a.seed(9123); b.seed(9123)
        obs = a.reset()
        assert np.array_equal(obs, b.reset())
        for step in range(205):
            actions = np.zeros((8, a.action_space.shape[0]), np.float32)
            x, y = a.step(actions), b.step(actions)
            for i in range(3):
                assert np.array_equal(x[i], y[i])
            if step == 199:
                assert x[2].all()
                assert all(i['TimeLimit.truncated'] and 'terminal_observation' in i for i in x[3])
                assert all(np.array_equal(i['terminal_observation'], j['terminal_observation']) for i, j in zip(x[3], y[3]))
        collector = dict(passed=True, same_seed_independent_of_process_grouping=True,
                         native_time_limit_and_terminal_observation=True)
    finally:
        a.close(); b.close()
    env = make_env('door')
    try:
        replay = DemonstrationReplay(4096, env.observation_space, env.action_space, device='cpu', n_envs=1,
            demo_path='/workspace/heechan-adroit/datasets/door-human-sparse-v2.npz', demo_seed=44)
        # Deliberately tagged online observations distinguish sources in every minibatch.
        obs = np.full((1, *env.observation_space.shape), 1234., np.float32)
        replay.add(obs, obs+1, np.zeros((1, env.action_space.shape[0]), np.float32),
                   np.array([0.], np.float32), np.array([True]), [{'success': False, 'TimeLimit.truncated': True}])
        for size in (256, 8192):
            batch = replay.sample(size)
            assert np.all(batch.observations.numpy()[1::2] == 1234)
            assert not np.any(batch.observations.numpy()[0::2] == 1234)
            assert np.all(batch.dones.numpy() == 0), 'Timeouts must bootstrap'
            for start in range(0, size, 256):
                assert np.sum(batch.observations.numpy()[start:start+256, 0] == 1234) == 128
        assert replay.first_online_success_at is None, 'Offline demonstrations are not online discoveries'
        result = dict(passed=True, collector=collector, replay=dict(passed=True,
            exact_demo_online_per_optimizer_batch=[128, 128], native_ibolt_32_update_sample=8192,
            timeouts_bootstrap=True, demonstrations_excluded_from_discovery=True))
        atomic(Path('/workspace/heechan-adroit/runtime-checks.json'), result)
        print(result, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
