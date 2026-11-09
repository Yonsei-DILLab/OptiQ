"""Frozen historical PointMaze native adapters; private dependency paths only."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import importlib.util
import sys
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from ..envs.replay import Replay
ROOT = Path(__file__).resolve().parent
class SpaceOnlyEnv(gym.Env):
    metadata = {}

    def __init__(self, observation_dim):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("SpaceOnlyEnv holds shapes; TaskBatch owns the dynamics")

    def step(self, action):
        raise RuntimeError("SpaceOnlyEnv holds shapes; TaskBatch owns the dynamics")

class SAC:
    method = "sac"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256, temperature=None, device="cuda"):
        from stable_baselines3 import SAC as SB3SAC
        from stable_baselines3.common.logger import configure
        self.model = SB3SAC("MlpPolicy", SpaceOnlyEnv(observation_dim), learning_rate=3e-4,
                            buffer_size=1_000_000, learning_starts=0,
                            batch_size=batch_size, tau=.005, gamma=.99,
                            train_freq=1, gradient_steps=1, ent_coef="auto",
                            policy_kwargs=dict(net_arch=[256, 256]),
                            seed=seed, device=device, verbose=0)
        self.model.set_logger(configure(str(folder / "learner"), ["csv"]))
        self.replay = Replay(seed=seed, observation_dim=observation_dim, device=device)
        self.model.replay_buffer = self.replay
        self.batch_size = batch_size
        self.budget = budget
        self.config = dict(algorithm="Stable-Baselines3 SAC", network=[256, 256],
                           actor_lr=3e-4, critic_lr=3e-4, entropy="automatic")

    @property
    def updates(self):
        return self.model._n_updates

    def act(self, obs, mode="policy"):
        action, _ = self.model.predict(np.asarray(obs, np.float32), deterministic=mode == "native")
        return np.clip(action, -1., 1.).astype(np.float32)

    def q(self, obs, actions):
        import torch
        with torch.no_grad():
            states = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.model.device)
            return torch.minimum(*self.model.critic(states, actions)).cpu().numpy().reshape(-1)

    critic_label = "minimum of two SAC critics"

    def update(self, steps):
        self.model.num_timesteps = steps
        self.model._current_progress_remaining = max(0., 1. - steps / self.budget)
        self.model.train(batch_size=self.batch_size, gradient_steps=1)
        return {key: float(value) for key, value in self.model.logger.name_to_value.items()
                if isinstance(value, (int, float, np.number)) and np.isfinite(value)}

    @contextmanager
    def evaluation_rng(self, seed):
        import torch
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if self.model.device.type == "cuda" else []):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            yield

    def save(self, folder, step, full=False):
        path = folder / f"policy_{step:09d}.zip"
        self.model.save(path)
        if full:
            np.savez_compressed(folder / f"replay_{step:09d}.npz",
                                **{key: value[:self.replay.size] for key, value in self.replay.data.items()},
                                position=self.replay.position, size=self.replay.size)
        return path

class TD3:
    """Stable-Baselines3 TD3 with native deterministic evaluation policy."""

    method = "td3"
    critic_label = "minimum of two TD3 critics"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256, temperature=None, device="cuda"):
        from stable_baselines3 import TD3 as SB3TD3
        from stable_baselines3.common.logger import configure

        self.model = SB3TD3(
            "MlpPolicy", SpaceOnlyEnv(observation_dim), learning_rate=3e-4,
            buffer_size=1_000_000, learning_starts=0, batch_size=batch_size,
            tau=.005, gamma=.99, train_freq=1, gradient_steps=1,
            policy_delay=2, target_policy_noise=.2, target_noise_clip=.5,
            policy_kwargs=dict(net_arch=[256, 256]), seed=seed,
            device=device, verbose=0,
        )
        self.model.set_logger(configure(str(folder / "learner"), ["csv"]))
        self.replay = Replay(seed=seed, observation_dim=observation_dim, device=device)
        self.model.replay_buffer = self.replay
        self.noise_rng = np.random.default_rng(seed + 991)
        self.batch_size = batch_size
        self.budget = budget
        self.config = dict(algorithm="Stable-Baselines3 TD3", network=[256, 256],
                           actor_lr=3e-4, critic_lr=3e-4, policy_delay=2,
                           target_policy_noise=.2, target_noise_clip=.5,
                           training_exploration_std=.1,
                           evaluation_policy="deterministic actor")

    @property
    def updates(self):
        return self.model._n_updates

    def act(self, obs, mode="policy"):
        actions, _ = self.model.predict(np.asarray(obs, np.float32), deterministic=True)
        if mode == "train":
            actions = actions + self.noise_rng.normal(0., .1, size=actions.shape)
        return np.clip(actions, -1., 1.).astype(np.float32)

    def q(self, obs, actions):
        import torch
        with torch.no_grad():
            states = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.model.device)
            return torch.minimum(*self.model.critic(states, actions)).cpu().numpy().reshape(-1)

    def update(self, steps):
        self.model.num_timesteps = steps
        self.model._current_progress_remaining = max(0., 1. - steps / self.budget)
        self.model.train(batch_size=self.batch_size, gradient_steps=1)
        return {key: float(value) for key, value in self.model.logger.name_to_value.items()
                if isinstance(value, (int, float, np.number)) and np.isfinite(value)}

    @contextmanager
    def evaluation_rng(self, seed):
        del seed
        yield

    def save(self, folder, step, full=False):
        path = folder / f"policy_{step:09d}.zip"
        self.model.save(path)
        if full:
            np.savez_compressed(folder / f"replay_{step:09d}.npz",
                                **{key: value[:self.replay.size] for key, value in self.replay.data.items()},
                                position=self.replay.position, size=self.replay.size)
        return path

class SQL:
    method = "sql"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256,
                 temperature=3., particles=16):
        from .sql_jax import SQLConfig, SQLLearner
        self.config = SQLConfig(hidden_dims=(256, 256), temperature=temperature,
                                kernel_particles=particles, value_particles=particles)
        self.learner = SQLLearner(observation_dim, 2, seed, self.config)
        self.replay = Replay(seed=seed, observation_dim=observation_dim)
        self.batch_size = batch_size
        self.action_key = jax.random.PRNGKey(seed + 1024)

    @property
    def updates(self):
        return int(self.learner.state.updates)

    def act(self, obs, mode="policy"):
        self.action_key, sample_key = jax.random.split(self.action_key)
        action = self.learner.sample_fn(self.learner.state.actor.params,
                                        np.asarray(obs, np.float32), sample_key)
        return np.clip(np.asarray(action), -1., 1.).astype(np.float32)

    def q(self, obs, actions):
        return np.asarray(self.learner.q_fn(self.learner.state.critic.params,
                                          np.asarray(obs, np.float32),
                                          np.asarray(actions, np.float32))).reshape(-1)

    critic_label = "live SQL soft Q"

    def update(self, steps):
        return {key: float(value) for key, value in self.learner.update(
            self.replay.batch(self.batch_size), iteration=self.updates).items()}

    @contextmanager
    def evaluation_rng(self, seed):
        key = self.action_key
        self.action_key = jax.random.PRNGKey(seed)
        try:
            yield
        finally:
            self.action_key = key

    def save(self, folder, step, full=False):
        path = folder / f"policy_{step:09d}.msgpack"
        self.learner.save(path)
        if full:
            np.savez_compressed(folder / f"replay_{step:09d}.npz",
                                **{key: value[:self.replay.size] for key, value in self.replay.data.items()},
                                position=self.replay.position, size=self.replay.size)
        return path

class MEOW:
    """Original MEOW CleanRL flow and Q/V identity with shared vector replay."""

    method = "meow"
    critic_label = "MEOW flow-derived Q"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256,
                 temperature=None, alpha=.2, device="auto"):
        import copy
        import torch

        upstream = ROOT / "vendor/meow/cleanrl"
        if not (upstream / "cleanrl/meow_continuous_action.py").exists():
            raise FileNotFoundError("initialize pinned gmm40-baseline/meow submodule")
        sys.path.insert(0, str(upstream))
        from cleanrl.meow_continuous_action import FlowPolicy

        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("MEOW alpha must be positive and finite")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in ("cuda", "cpu"):
            raise ValueError("MEOW device must be cuda, cpu, or auto")
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.policy = FlowPolicy(alpha=alpha, sigma_max=-.3, sigma_min=-5.,
                                 action_sizes=2, state_sizes=observation_dim,
                                 device=self.device).to(self.device)
        self.target = copy.deepcopy(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.replay = Replay(seed=seed, observation_dim=observation_dim)
        self.batch_size = batch_size
        self.count = 0
        self.config = dict(algorithm="MEOW CleanRL flow Q regression",
                           upstream_commit="b786d27aa9b03e4242ee8904ff884b21fe65e2f7",
                           alpha=alpha, sigma_min=-5., sigma_max=-.3,
                           device=device,
                           q_lr=1e-3, tau=.005, grad_clip=30,
                           evaluation_policy="direct flow sample")

    @property
    def updates(self):
        return self.count

    def act(self, obs, mode="policy"):
        import torch
        del mode
        self.policy.eval()
        with torch.no_grad():
            actions, _ = self.policy.sample(num_samples=len(obs), obs=obs, deterministic=False)
        return np.clip(actions.cpu().numpy(), -1., 1.).astype(np.float32)

    def q(self, obs, actions):
        import torch
        self.policy.eval()
        with torch.no_grad():
            states = torch.as_tensor(np.asarray(obs, np.float32), device=self.device)
            actions = torch.as_tensor(np.asarray(actions, np.float32), device=self.device)
            q, _ = self.policy.get_qv(states, actions)
        return q.cpu().numpy().reshape(-1)

    def update(self, steps):
        import torch
        import torch.nn.functional as F
        del steps
        batch = self.replay.batch(self.batch_size)
        tensor = lambda key: torch.as_tensor(batch[key], device=self.device)
        obs, action, reward = tensor("observations"), tensor("actions"), tensor("rewards")
        next_obs, done = tensor("next_observations"), tensor("terminals")
        with torch.no_grad():
            self.target.eval()
            duplicated_v = self.target.get_v(torch.cat((next_obs, next_obs), dim=0))
            target_v = torch.minimum(*duplicated_v.chunk(2, dim=0)).flatten()
            target_q = reward + (1. - done) * .99 * target_v
        self.policy.train()
        duplicated_q, _ = self.policy.get_qv(torch.cat((obs, obs), dim=0),
                                              torch.cat((action, action), dim=0))
        loss = F.mse_loss(duplicated_q.flatten(), torch.cat((target_q, target_q)))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite MEOW Q regression loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 30., error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for online, target in zip(self.policy.parameters(), self.target.parameters()):
                target.lerp_(online, .005)
        self.count += 1
        return {"q_loss": float(loss), "q_mean": float(duplicated_q.detach().mean()),
                "td_target_mean": float(target_q.mean()), "alpha": float(self.policy.alpha)}

    @contextmanager
    def evaluation_rng(self, seed):
        import torch
        cuda_devices = [torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            yield

    def save(self, folder, step, full=False):
        import torch
        path = folder / f"policy_{step:09d}.pt"
        torch.save(dict(policy=self.policy.state_dict(), target=self.target.state_dict(),
                        optimizer=self.optimizer.state_dict(), updates=self.count), path)
        if full:
            np.savez_compressed(folder / f"replay_{step:09d}.npz",
                                **{key: value[:self.replay.size] for key, value in self.replay.data.items()},
                                position=self.replay.position, size=self.replay.size)
        return path

class MFPO:
    method = "mfpo"
    critic_label = "MFPO clipped-double distributional value"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256,
                 temperature=None, target_entropy_coeff=-.5):
        # Frozen training sources contain the exact pinned submodule files but
        # deliberately omit their Git metadata. Load the author's config from
        # that snapshot; its parent gitlink and source sidecar pin the revision.
        config_path = ROOT / "vendor/mfpo/configs/mfpo_config.py"
        if not config_path.is_file():
            raise FileNotFoundError("pinned MFPO source is absent from training snapshot")
        spec = importlib.util.spec_from_file_location("maze_mfpo_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        self.config = config_module.get_config().to_dict()
        self.config.pop("model_cls")
        self.config["target_entropy_coeff"] = target_entropy_coeff
        self.config["upstream_commit"] = "d8b3977d29d4ef2d315e871337e5826f2eb79eb2"
        sys.path.insert(0, str(ROOT / "vendor/mfpo"))
        from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
        descriptor = SpaceOnlyEnv(observation_dim)
        descriptor.observation_space.seed(seed)
        descriptor.action_space.seed(seed)
        training_config = {key: value for key, value in self.config.items()
                           if key != "upstream_commit"}
        self.agent = MeanFlowLearner.create(
            seed, descriptor.observation_space, descriptor.action_space, **training_config)
        self.replay = Replay(seed=seed, observation_dim=observation_dim)
        self.batch_size = batch_size
        self.count = 0

        @jax.jit
        def sample(agent, obs):
            keys = jax.random.split(agent.rng, len(obs) + 1)
            actions = jax.vmap(
                lambda state, key: agent.replace(rng=key).sample_actions(state)[0]
            )(obs, keys[1:])
            return actions, agent.replace(rng=keys[0])

        self.sample_fn = sample
        self.value_fn = jax.jit(lambda agent, obs, action: agent.calc_value(obs, action))

    @property
    def updates(self):
        return self.count

    def act(self, obs, mode="policy"):
        actions, self.agent = self.sample_fn(self.agent, np.asarray(obs, np.float32))
        return np.clip(np.asarray(actions), -1., 1.).astype(np.float32)

    def q(self, obs, actions):
        return np.asarray(self.value_fn(self.agent, np.asarray(obs, np.float32),
                                        np.asarray(actions, np.float32))).reshape(-1)

    def update(self, steps):
        sampled = self.replay.batch(self.batch_size)
        terminal = sampled["terminals"]
        data = dict(observations=sampled["observations"], actions=sampled["actions"],
                    next_observations=sampled["next_observations"],
                    rewards=sampled["rewards"], masks=1. - terminal, dones=terminal)
        self.agent, info = self.agent.update(data, utd_ratio=1)
        self.count += 1
        return {key: float(value) for key, value in info.items()
                if np.asarray(value).size == 1 and np.isfinite(value)}

    @contextmanager
    def evaluation_rng(self, seed):
        agent = self.agent
        self.agent = self.agent.replace(rng=jax.random.PRNGKey(seed))
        try:
            yield
        finally:
            self.agent = agent

    def save(self, folder, step, full=False):
        import flax.serialization
        path = folder / f"policy_{step:09d}.msgpack"
        path.write_bytes(flax.serialization.to_bytes(self.agent))
        if full:
            np.savez_compressed(folder / f"replay_{step:09d}.npz",
                                **{key: value[:self.replay.size] for key, value in self.replay.data.items()},
                                position=self.replay.position, size=self.replay.size)
        return path
