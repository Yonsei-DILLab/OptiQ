"""GMM40 fixed-energy and online-navigation adapters for the JAX SQL port."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import jax
import numpy as np

from .sql_jax import SQLConfig, SQLLearner, UPSTREAM_COMMIT
from .target import SCALE


def config_from_args(args):
    return SQLConfig(hidden_dims=(args.width,) * args.depth, temperature=args.temperature,
                     kernel_particles=args.sql_kernel_particles,
                     kernel_update_ratio=args.sql_kernel_update_ratio,
                     value_particles=args.sql_value_particles,
                     target_update_interval=args.sql_target_update_interval)


def metadata(args):
    try:
        commit = subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),
                                          'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(sql=asdict(config_from_args(args)), actor_hidden_dims=[args.width] * args.depth,
                source_git_commit=commit,
                actor_learning_rate=3e-4, sql_upstream_commit=UPSTREAM_COMMIT,
                sql_implementation="JAX/Flax amortized SVGD; original bounded-action score",
                sql_policy_gradient_reduction="sum over batch and updated particles; mean over fixed particles",
                sql_target_update_clock="zero-based environment step; hard copy after update",
                sql_optimizer="TF1-compatible Adam; no gradient clipping",
                latent_distribution="fresh standard normal, dimension=action_dim")


class SQL:
    def __init__(self, target, seed=0, batch=256, config=SQLConfig()):
        if batch <= 0:
            raise ValueError("batch must be positive")
        self.batch, self.config = batch, config
        self.learner = SQLLearner(1, 2, seed, config,
                                 fixed_q=lambda obs, a: target.jax_log_prob(SCALE * a),
                                 checkpoint_context=dict(batch=batch, scale=SCALE,
                                     target_sha256=hashlib.sha256(json.dumps(getattr(target,'metadata',None),sort_keys=True).encode()).hexdigest()))

    @property
    def updates(self):
        return int(self.learner.state.updates)

    @property
    def state(self):
        return self.learner.state.actor

    def advance(self, count):
        info = {k: float(v) for k, v in self.learner.advance(count, self.batch).items()}
        fixed = self.config.kernel_particles - int(self.config.kernel_particles * self.config.kernel_update_ratio)
        return {**info, "Q_evaluations": self.updates * self.batch * fixed}

    def evaluate_samples(self, n, seed):
        samples = self.learner.sample_fn(self.state.params, np.zeros((n, 1), np.float32), jax.random.PRNGKey(seed))
        return SCALE * np.asarray(samples), None, {}

    def save(self, path):
        self.learner.save(path)

    def restore(self, path):
        self.learner.restore(path)


class SQLOnline:
    """Replay and RNG ownership for normalized-action Gymnasium environments."""
    def __init__(self, env, seed, folder=None, batch=256, config=SQLConfig(), capacity=1000000):
        if batch <= 0 or capacity <= 0:
            raise ValueError("batch and capacity must be positive")
        if not (np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)):
            raise ValueError("SQL expects actions normalized to [-1, 1]")
        self.learner = SQLLearner(env.observation_space.shape[0], env.action_space.shape[0], seed, config)
        self.batch, self.capacity = batch, capacity
        self.rng = np.random.default_rng(seed)
        self.eval_key = None
        self.size, self.position, self.env_steps = 0, 0, 0
        od, ad = self.learner.observation_dim, self.learner.action_dim
        self.replay = {key: np.empty(shape, np.float32) for key, shape in dict(
            observations=(capacity, od), actions=(capacity, ad), rewards=(capacity,),
            next_observations=(capacity, od), terminals=(capacity,)).items()}

    @property
    def updates(self):
        return int(self.learner.state.updates)

    def act(self, obs):
        if self.eval_key is None:
            key, sample_key = jax.random.split(self.learner.state.key)
            self.learner.state = self.learner.state.replace(key=key)
        else:
            self.eval_key, sample_key = jax.random.split(self.eval_key)
        return np.asarray(self.learner.sample_fn(self.learner.state.actor.params, np.asarray(obs, np.float32), sample_key))

    def q(self, obs, actions):
        return np.asarray(self.learner.q_fn(self.learner.state.critic.params, np.asarray(obs, np.float32), np.asarray(actions, np.float32)))

    def store(self, s, a, r, ns, done):
        for key, value in dict(observations=s, actions=a, rewards=r, next_observations=ns, terminals=done).items():
            self.replay[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.env_steps += 1

    def update(self):
        if not self.size:
            raise ValueError("Cannot update from empty replay")
        indices = self.rng.integers(self.size, size=self.batch)
        info = self.learner.update({k: v[indices] for k, v in self.replay.items()}, iteration=self.env_steps - 1)
        return {k: float(v) for k, v in info.items()}

    def save(self, path):
        self.learner.save(path)
        with path.with_suffix(".replay.pkl").open("wb") as f:
            pickle.dump(dict(replay={k: v[:self.size] for k, v in self.replay.items()},
                             size=self.size, position=self.position, env_steps=self.env_steps,
                             capacity=self.capacity, batch=self.batch, rng=self.rng.bit_generator.state), f)

    def restore(self, path):
        # Only restore trusted local checkpoints; the replay sidecar uses pickle.
        with path.with_suffix(".replay.pkl").open("rb") as f:
            data = pickle.load(f)
        if data["capacity"] != self.capacity or data["batch"] != self.batch:
            raise ValueError("SQL replay capacity/batch does not match checkpoint")
        self.learner.restore(path)
        self.size, self.position, self.env_steps = data["size"], data["position"], data["env_steps"]
        for k, v in data["replay"].items():
            self.replay[k][:self.size] = v
        self.rng.bit_generator.state = data["rng"]
        self.eval_key = None
