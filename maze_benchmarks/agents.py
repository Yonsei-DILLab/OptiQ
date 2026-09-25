"""Online adapters around existing OptiQ, SAC and JAX SQL implementations."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import importlib.util
import sys

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from .replay import Replay


ROOT = Path(__file__).resolve().parents[1]


class SpaceOnlyEnv(gym.Env):
    metadata = {}

    def __init__(self, observation_dim):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("SpaceOnlyEnv holds shapes; TaskBatch owns the dynamics")

    def step(self, action):
        raise RuntimeError("SpaceOnlyEnv holds shapes; TaskBatch owns the dynamics")


class OptiQ:
    method = "optiq"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256, temperature=3.):
        from omegaconf import OmegaConf
        from stable_baselines3.common.logger import configure

        trainer_path = ROOT / "analysis_tools/experiments/20260921_gmm_trg_sweep/train.py"
        spec = importlib.util.spec_from_file_location("maze_optiq_trainer", trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        cfg = trainer.compose_config([
            "benchmark=ant", f"seed={seed}",
            f"alg.actor.temperature={temperature}",
            "alg.actor.mean_output_init_scale=0.0001",
            "alg.actor.log_std_min=-5.0", "alg.actor.log_std_max=-1.0",
            "alg.actor.initial_log_std=-1.0",
            "alg.actor.teacher_std_floor=0.006737946999085467",
            "dacer.enabled=false", f"output_root={folder}",
        ], allowed_policy_samples=64)
        cfg.total_steps = budget
        cfg.alg.learning_starts = 0
        cfg.alg.actor.learning_starts = 0
        cfg.alg.batch_size = batch_size
        cfg.alg.buffer_size = 1_000_000
        cfg.alg.gamma = .99
        cfg.alg.tau = .005
        cfg.alg.policy_tau = 1.
        cfg.alg.policy_delay = 1
        cfg.alg.critic.n_atoms = 1
        cfg.alg.critic.backup_mode = "td"
        cfg.alg.critic.crossq_style = False
        cfg.alg.critic.hs = [256, 256]
        cfg.alg.actor.hidden_dims = [256, 256]
        cfg.alg.actor.include_anchor = False
        cfg.alg.actor.density_correction = True
        cfg.alg.actor.density_beta = 1.
        cfg.alg.actor.adaptive_density_beta = False
        cfg.alg.actor.distillation_loss = "direct_gmm_nll"
        cfg.alg.actor.num_policy_samples = 64
        cfg.alg.actor.proposals_per_policy_sample = 1
        cfg.alg.actor.proposal_sampling_mode = "exact"
        cfg.alg.optimizer.lr_actor = 3e-4
        cfg.alg.optimizer.lr_critic = 3e-4
        cfg.wandb.activate = False
        self.model = trainer.runner.OptiQDIME("MlpPolicy", SpaceOnlyEnv(observation_dim), cfg=cfg,
                                              model_save_path=None, save_every_n_steps=budget)
        self.model.set_logger(configure(str(folder / "learner"), ["csv"]))
        self.replay = Replay(seed=seed, observation_dim=observation_dim)
        self.model.replay_buffer = self.replay
        self.model._total_timesteps = budget
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.batch_size = batch_size
        self.budget = budget

    @property
    def updates(self):
        return self.model._n_updates

    def act(self, obs, mode="policy"):
        policy = self.model.policy
        policy.reset_noise()
        action = policy.sample_action(policy.actor_state, jnp.asarray(obs), policy.noise_key,
                                      deterministic=False,
                                      sample_conditional_noise=mode != "mu_only")
        return np.clip(np.asarray(action), -1., 1.).astype(np.float32)

    def q(self, obs, actions):
        state = self.model.policy.qf_state
        values = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            jnp.asarray(obs, jnp.float32), jnp.asarray(actions, jnp.float32),
            rngs={"dropout": jax.random.PRNGKey(31415)}, train=False,
        )
        return np.asarray(values).mean(axis=0).reshape(-1)

    critic_label = "mean of two live OptiQ critics"

    def update(self, steps):
        model = self.model
        model.num_timesteps = steps
        model._current_progress_remaining = max(0., 1. - steps / self.budget)
        model.train(batch_size=self.batch_size, gradient_steps=1)
        return {key: float(value) for key, value in model.logger.name_to_value.items()
                if isinstance(value, (int, float, np.number)) and np.isfinite(value)}

    @contextmanager
    def evaluation_rng(self, seed):
        policy = self.model.policy
        key, noise_key = policy.key, policy.noise_key
        policy.key, policy.noise_key = jax.random.PRNGKey(seed), jax.random.PRNGKey(seed + 1)
        try:
            yield
        finally:
            policy.key, policy.noise_key = key, noise_key

    def save(self, folder, step, full=False):
        import flax.serialization
        policy = self.model.policy
        state = dict(actor=policy.actor_state, critic=policy.qf_state,
                     target_actor=policy.target_actor_state,
                     target_critic_params=policy.qf_state.target_params)
        path = folder / f"policy_{step:09d}.msgpack"
        path.write_bytes(flax.serialization.to_bytes(state))
        if full:
            np.savez_compressed(folder / f"replay_{step:09d}.npz",
                                **{key: value[:self.replay.size] for key, value in self.replay.data.items()},
                                position=self.replay.position, size=self.replay.size)
        return path


class SAC:
    method = "sac"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256, temperature=None):
        from stable_baselines3 import SAC as SB3SAC
        from stable_baselines3.common.logger import configure
        self.model = SB3SAC("MlpPolicy", SpaceOnlyEnv(observation_dim), learning_rate=3e-4,
                            buffer_size=1_000_000, learning_starts=0,
                            batch_size=batch_size, tau=.005, gamma=.99,
                            train_freq=1, gradient_steps=1, ent_coef="auto",
                            policy_kwargs=dict(net_arch=[256, 256]),
                            seed=seed, device="cuda", verbose=0)
        self.model.set_logger(configure(str(folder / "learner"), ["csv"]))
        self.replay = Replay(seed=seed, observation_dim=observation_dim, device="cuda")
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
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
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


class SQL:
    method = "sql"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256, temperature=3.):
        from gmm40.sql_jax import SQLConfig, SQLLearner
        self.config = SQLConfig(hidden_dims=(256, 256), temperature=temperature)
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


class MFPO:
    method = "mfpo"
    critic_label = "MFPO clipped-double distributional value"

    def __init__(self, seed, folder, budget, observation_dim, batch_size=256, temperature=None):
        from antmaze_experiments.dependencies import load_mfpo_config
        self.config = load_mfpo_config(ROOT).to_dict()
        self.config.pop("model_cls")
        sys.path.insert(0, str(ROOT / "gmm40-baseline/MFPO"))
        from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
        descriptor = SpaceOnlyEnv(observation_dim)
        descriptor.observation_space.seed(seed)
        descriptor.action_space.seed(seed)
        self.agent = MeanFlowLearner.create(
            seed, descriptor.observation_space, descriptor.action_space, **self.config)
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
