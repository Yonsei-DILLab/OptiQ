"""Exercise actual SB3 collection, warmup and a partial last rollout with zero optimizer work."""
import json
from pathlib import Path
import tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.save_util import load_from_zip_file
from rebuttal_tools.fetch_her_episode_collection import configure_episode_collection, run_optimizer_chunks


class CounterEnv(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (1,), np.float32)
    action_space = spaces.Box(-1, 1, (4,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.time += 1
        return np.array([self.time], np.float32), -1., False, self.time == 50, {}


def run():
    env = DummyVecEnv([CounterEnv for _ in range(256)])
    try:
        model = SAC('MlpPolicy', env, seed=5, device='cpu', verbose=0,
                    policy_kwargs=dict(net_arch=[8, 8]), learning_starts=12800,
                    buffer_size=40000, batch_size=256, gradient_steps=32)
        initial = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
        calls, chunks, collection = [], [], []
        model.model_save_path = 'checkpoint-sentinel'

        def native_train(batch_size, gradient_steps):
            assert batch_size == 256 and gradient_steps == 32
            assert model.model_save_path is None
            chunks.append(gradient_steps)
            model._n_updates += gradient_steps

        def train(batch_size, gradient_steps):
            calls.append((model.num_timesteps, gradient_steps))
            run_optimizer_chunks(model, native_train, batch_size, gradient_steps)
            assert model._n_updates == (model.num_timesteps - 12800) // 8

        def callback(locals_, _globals):
            collection.append((model.num_timesteps, model._n_updates))
            return True

        model.train = train
        total = 33792  # warmup + one full rollout + a 32-vector-step last rollout
        configure_episode_collection(model, total)
        model.learn(total_timesteps=total, callback=callback, log_interval=100000)
        assert model.num_timesteps == total
        assert calls == [(25600, 1600), (33792, 1024)], calls
        assert chunks == [32] * 82 and model._n_updates == 2624
        assert all(updates == (0 if step <= 25600 else 1600) for step, updates in collection)
        assert model.model_save_path == 'checkpoint-sentinel'
        assert all(np.array_equal(tensor.numpy(), initial[name].numpy())
                   for name, tensor in model.policy.state_dict().items()), 'Probe accidentally optimized parameters'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.zip'
            model.save(path, exclude=['train', 'collect_rollouts'])
            data, _, _ = load_from_zip_file(path, device='cpu')
            assert 'train' not in data and 'collect_rollouts' not in data
        full_budget = 5000192
        assert (full_budget - 12800) // 8 == 623424
        return dict(passed=True, optimizer_updates_executed=0,
                    simulated_update_counter=model._n_updates,
                    actual_environment_steps=model.num_timesteps,
                    native_chunk_size=32, training_calls=calls,
                    no_mid_episode_training=True, exact_partial_last_rollout=True,
                    expected_updates_at_5m=623424,
                    checkpoint_excludes_bound_collection_closure=True)
    finally:
        env.close()


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
