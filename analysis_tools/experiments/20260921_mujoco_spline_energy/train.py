"""Online MuJoCo training for the unified spline energy circuit."""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from flax import serialization
from flax.training.train_state import TrainState
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from circuit import (ConditionalSplineCircuit, log_prob_from_output,
                     q_from_output, sample_action, sample_from_output)
from algorithms.spline_energy.energy_bellman import relative_energy_loss
from algorithms.spline_energy.raw_energy import ConditionalRawEnergyCircuit

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4", "Humanoid-v4"]


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, seed):
        self.capacity, self.position, self.size = capacity, 0, 0
        self.obs = np.empty((capacity, obs_dim), np.float32)
        self.next_obs = np.empty((capacity, obs_dim), np.float32)
        self.actions = np.empty((capacity, action_dim), np.float32)
        self.rewards = np.empty((capacity,), np.float32)
        self.not_terminal = np.empty((capacity,), np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.position
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.not_terminal[i] = next_obs, 0.0 if terminated else 1.0
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        i = self.rng.integers(self.size, size=batch_size)
        return {"obs": self.obs[i], "actions": self.actions[i],
                "rewards": self.rewards[i], "next_obs": self.next_obs[i],
                "not_terminal": self.not_terminal[i]}


def make_update(model, temperature, gamma, tau, huber_delta,
                loss_kind="huber", energy_scale=10., energy_tail_start=4.,
                backup_mode="soft"):
    if loss_kind not in {"huber", "relative_energy", "mse"}:
        raise ValueError(loss_kind)
    if backup_mode not in {"soft", "td"}:
        raise ValueError(backup_mode)
    if isinstance(model, ConditionalRawEnergyCircuit) and model.temperature != temperature:
        raise ValueError("Circuit and Bellman temperatures must agree")
    @jax.jit
    def update(state, target_params, batch, backup_key=None):
        target_output = model.apply({"params": target_params}, batch["next_obs"])
        if backup_mode == "td":
            if backup_key is None:
                raise ValueError("Plain TD needs an explicit independent backup RNG key")
            # Direct GMM/TRG samples from the current policy and evaluates a
            # delayed critic. Here both are views of the same circuit.
            current_next_output = model.apply({"params": state.params}, batch["next_obs"])
            next_actions = jax.lax.stop_gradient(sample_from_output(current_next_output, backup_key))
            bootstrap = q_from_output(target_output, next_actions, temperature)
        else:
            bootstrap = target_output["value"]
        target = jax.lax.stop_gradient(
            batch["rewards"] + gamma * batch["not_terminal"] * bootstrap)
        def loss_fn(params):
            output = model.apply({"params": params}, batch["obs"])
            logp = log_prob_from_output(output, batch["actions"])
            q = q_from_output(output, batch["actions"], temperature)
            if loss_kind == "relative_energy":
                loss = relative_energy_loss(q - target, energy_scale, energy_tail_start).mean()
            elif loss_kind == "mse":
                loss = jnp.square(q - target).mean()
            else:
                loss = optax.huber_loss(q, target, delta=huber_delta).mean()
            return loss, (q, output["value"], logp)
        (loss, (q, value, logp)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        target_params = optax.incremental_update(state.params, target_params, tau)
        metrics = {"train/loss": loss, "train/q_mean": q.mean(),
                   "train/data_log_prob": logp.mean(),
                   "train/target_mean": target.mean(),
                   "train/abs_td": jnp.abs(q - target).mean(),
                   "train/grad_norm_preclip": grad_norm}
        if backup_mode == "soft":
            metrics["train/v_mean"] = value.mean()
        else:
            # alpha log Z is not the ordinary expected policy value.
            metrics["train/alpha_log_partition_mean"] = value.mean()
            metrics["train/bootstrap_q_mean"] = bootstrap.mean()
        if loss_kind == "relative_energy":
            # Sample diagnostics only: none is a global Bellman certificate.
            metrics.update({
                "train/sampled_abs_td_max": jnp.max(jnp.abs(q - target)),
                "train/sampled_td_over_fraction": jnp.mean(q > target),
                "train/energy_tail_fraction": jnp.mean(
                    q - target > energy_scale * energy_tail_start),
            })
        return state, target_params, metrics
    return update


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git_value(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def to_environment_action(action, low, high):
    """Map the circuit's [-1,1] coordinate to each MuJoCo action box."""
    return low + 0.5 * (np.asarray(action) + 1.0) * (high - low)


def to_normalized_action(action, low, high):
    return np.clip(2.0 * (np.asarray(action) - low) / (high - low) - 1.0, -1.0, 1.0)


def evaluate(env_name, params, model, seed, episodes):
    env = gym.make(env_name)
    key = jax.random.PRNGKey(8_000_000 + seed)
    returns, lengths = [], []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=1_000_000 + seed * 100 + episode)
            total, length, done = 0.0, 0, False
            while not done:
                key, action_key = jax.random.split(key)
                normalized_action = np.asarray(sample_action(
                    params, jnp.asarray(obs, dtype=jnp.float32)[None], action_key, model=model)[0])
                action = to_environment_action(normalized_action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward); length += 1; done = terminated or truncated
            returns.append(total); lengths.append(length)
    finally:
        env.close()
    return {"eval/mean_reward": float(np.mean(returns)),
            "eval/std_reward": float(np.std(returns)),
            "eval/mean_length": float(np.mean(lengths)),
            "eval/returns": returns, "eval/lengths": lengths}


def main():
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--profile", choices=["legacy", "gmm_reference"], default="legacy")
    selected, _ = profile_parser.parse_known_args()
    reference = selected.profile == "gmm_reference"
    parser = argparse.ArgumentParser(parents=[profile_parser])
    parser.add_argument("--env", choices=ENVS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=64 if reference else 16)
    parser.add_argument("--knots", type=int, default=129 if reference else 33)
    parser.add_argument("--parameterization", choices=["normalized_value", "raw_energy"],
                        default="raw_energy" if reference else "normalized_value")
    parser.add_argument("--backup-mode", choices=["soft", "td"], default="td" if reference else "soft")
    parser.add_argument("--grad-clip", type=float, default=0. if reference else 10.,
                        help="0 disables clipping, matching GMM40 and Direct GMM/TRG")
    parser.add_argument("--temperature", type=float, default=.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--tau", type=float, default=.005)
    parser.add_argument("--huber-delta", type=float, default=10.)
    parser.add_argument("--loss-kind", choices=["huber", "relative_energy", "mse"],
                        default="mse" if reference else "huber")
    parser.add_argument("--energy-scale", type=float, default=10.)
    parser.add_argument("--energy-tail-start", type=float, default=4.)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.grad_clip < 0 or not np.isfinite(args.grad_clip):
        raise ValueError("grad-clip must be finite and nonnegative")
    if reference and (args.parameterization != "raw_energy" or args.loss_kind != "mse"
                      or args.backup_mode != "td"):
        raise ValueError("gmm_reference requires raw energy + ordinary TD + MSE")
    if args.total_steps <= args.warmup or args.eval_interval <= 0:
        raise ValueError("Training must extend beyond warmup with positive evaluation interval")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    args.output.mkdir(parents=True, exist_ok=False)
    env = gym.make(args.env)
    env.action_space.seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    if not (np.isfinite(env.action_space.low).all() and np.isfinite(env.action_space.high).all()
            and np.all(env.action_space.high > env.action_space.low)):
        raise ValueError("Circuit requires finite box action bounds")
    obs_dim, action_dim = int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape))
    if args.parameterization == "raw_energy":
        model = ConditionalRawEnergyCircuit(action_dim=action_dim, rank=args.rank, knots=args.knots,
                                            temperature=args.temperature, initialization_seed=args.seed)
    else:
        model = ConditionalSplineCircuit(action_dim=action_dim, rank=args.rank, knots=args.knots)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.asarray(obs, dtype=jnp.float32)[None])["params"]
    optimizer = (optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adam(args.learning_rate))
                 if args.grad_clip else optax.adam(args.learning_rate))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    target_params = params
    update = make_update(model, args.temperature, args.gamma, args.tau, args.huber_delta,
                         args.loss_kind, args.energy_scale, args.energy_tail_start, args.backup_mode)
    backup_key = jax.random.PRNGKey(args.seed + 9_000_000)
    replay = ReplayBuffer(args.buffer_size, obs_dim, action_dim, args.seed + 101)
    source_root = Path(__file__).resolve().parents[3]
    command = sys.argv
    config = vars(args).copy(); config["output"] = str(config["output"])
    algorithm = ("spline_energy_gmm_reference" if reference else
                 ("spline_energy" if args.loss_kind == "huber" else f"spline_energy_{args.loss_kind}"))
    config.update(algorithm=algorithm, algorithm_display_name="Spline Energy Circuit",
                  model_equation=("Q=alpha*log_F; pi=F/integral_F; alpha_log_Z_derived"
                                  if args.parameterization == "raw_energy" else "Q=V+alpha*log_pi"),
                  bellman_target=("r+gamma*(1-terminal)*Q_target(s_next,a_current_policy)"
                                  if args.backup_mode == "td" else "r+gamma*(1-terminal)*alpha_log_Z_target"),
                  reference_direct_commit=("30f4db1cf9929974dacbdbca7c9c5abd5c1bd346" if reference else None),
                  policy=f"rank-{args.rank} mixture of products of normalized positive linear splines",
                  hidden_dims=[256, 256], updates_per_step=1, behavior_uniform_probability=0.,
                  time_limit_bootstrap=True, observation_dim=obs_dim, action_dim=action_dim,
                  trainable_parameters=sum(x.size for x in jax.tree_util.tree_leaves(params)),
                  git_commit=git_value(source_root, "rev-parse", "HEAD"),
                  git_branch=git_value(source_root, "branch", "--show-current"),
                  git_dirty=bool(git_value(source_root, "status", "--porcelain")),
                  command=command, hostname=platform.node(), slurm_job_id=os.getenv("SLURM_JOB_ID"),
                  gpu=os.popen("nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader").read().strip(),
                  packages={p: importlib.metadata.version(p) for p in
                            ["jax", "jaxlib", "flax", "optax", "numpy", "gymnasium", "mujoco", "wandb"]})
    atomic_json(args.output / "config.json", config)
    run = wandb.init(entity=os.getenv("WANDB_ENTITY"),
                     project=os.getenv("WANDB_PROJECT", "OptiQ-MuJoCo-Spline-Energy"),
                     group=os.getenv("WANDB_RUN_GROUP", "spline_energy"),
                     job_type="train" if args.total_steps == 1_000_000 else "smoke",
                     name=f"{args.env}-{algorithm}-s{args.seed}", mode=args.wandb_mode,
                     dir=str(args.output), config=config,
                     id=hashlib.sha256(str(args.output).encode()).hexdigest()[:12], resume="never",
                     settings=wandb.Settings(init_timeout=180))
    run.define_metric("env_steps"); run.define_metric("*", step_metric="env_steps")
    (args.output / "wandb.json").write_text(json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n")
    history_path = args.output / "evaluations.jsonl"
    episode_returns = deque(maxlen=20); episode_return = 0.; episode_length = 0
    train_seconds = 0.; last_metrics = {}; started = time.monotonic()
    try:
        initial = evaluate(args.env, state.params, model, args.seed, args.eval_episodes)
        initial.update(env_steps=0, training_seconds=0.)
        with history_path.open("w") as f:
            f.write(json.dumps(initial) + "\n"); f.flush()
            run.log({k: v for k, v in initial.items() if not isinstance(v, list)}, step=0)
            for env_step in range(1, args.total_steps + 1):
                if env_step <= args.warmup:
                    environment_action = env.action_space.sample()
                    action = to_normalized_action(environment_action,
                                                  env.action_space.low, env.action_space.high)
                else:
                    key, action_key = jax.random.split(key)
                    action = np.asarray(sample_action(state.params,
                        jnp.asarray(obs, dtype=jnp.float32)[None], action_key, model=model)[0])
                    environment_action = to_environment_action(
                        action, env.action_space.low, env.action_space.high)
                next_obs, reward, terminated, truncated, _ = env.step(environment_action)
                replay.add(np.asarray(obs, np.float32), np.asarray(action, np.float32),
                           reward, np.asarray(next_obs, np.float32), terminated)
                episode_return += float(reward); episode_length += 1; obs = next_obs
                if terminated or truncated:
                    episode_returns.append(episode_return)
                    obs, _ = env.reset()
                    episode_return, episode_length = 0., 0
                if env_step > args.warmup:
                    batch = {k: jnp.asarray(v) for k, v in replay.sample(args.batch_size).items()}
                    begin = time.monotonic()
                    if args.backup_mode == "td":
                        backup_key, step_backup_key = jax.random.split(backup_key)
                        state, target_params, metrics = update(state, target_params, batch, step_backup_key)
                    else:
                        state, target_params, metrics = update(state, target_params, batch)
                    jax.block_until_ready(state.params)
                    train_seconds += time.monotonic() - begin
                    last_metrics = {k: float(v) for k, v in metrics.items()}
                    if not all(np.isfinite(v) for v in last_metrics.values()):
                        raise FloatingPointError(last_metrics)
                if env_step % 1000 == 0:
                    payload = dict(last_metrics, env_steps=env_step,
                        **{"rollout/return20": float(np.mean(episode_returns)) if episode_returns else 0.,
                           "time/training_seconds": train_seconds,
                           "time/wall_seconds": time.monotonic() - started})
                    run.log(payload, step=env_step)
                    atomic_json(args.output / "latest.json", payload)
                    print(json.dumps(payload), flush=True)
                if env_step % args.eval_interval == 0 or env_step == args.total_steps:
                    result = evaluate(args.env, state.params, model, args.seed, args.eval_episodes)
                    result.update(env_steps=env_step, training_seconds=train_seconds)
                    f.write(json.dumps(result) + "\n"); f.flush()
                    run.log({k: v for k, v in result.items() if not isinstance(v, list)}, step=env_step)
                    run.summary.update({k: v for k, v in result.items() if not isinstance(v, list)})
                if args.checkpoint_interval and (env_step % args.checkpoint_interval == 0 or env_step == args.total_steps):
                    path = args.output / f"checkpoint_{env_step:07d}.msgpack"
                    checkpoint = {"state": state, "target_params": target_params,
                                  "key": key, "env_steps": env_step}
                    if args.backup_mode == "td":
                        checkpoint["backup_key"] = backup_key
                    path.write_bytes(serialization.to_bytes(checkpoint))
        final_rows = [json.loads(x) for x in history_path.read_text().splitlines()]
        xs = np.asarray([r["env_steps"] for r in final_rows], np.float64)
        ys = np.asarray([r["eval/mean_reward"] for r in final_rows], np.float64)
        auc = float(np.trapz(ys, xs) / max(xs[-1], 1.))
        complete = {"completed": True, "env_steps": args.total_steps, "optimizer_steps": int(state.step),
                    "eval_return_auc": auc, "final_eval_mean_reward": float(ys[-1]),
                    "wandb_url": run.url, "finished_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(args.output / "COMPLETE.json", complete)
        run.summary.update(complete)
        artifact = wandb.Artifact(f"spline-energy-{run.id}", type="mujoco-run")
        for path in [args.output / "config.json", history_path, args.output / "COMPLETE.json",
                     args.output / f"checkpoint_{args.total_steps:07d}.msgpack"]:
            if path.exists(): artifact.add_file(str(path))
        run.log_artifact(artifact)
    finally:
        env.close(); run.finish()


if __name__ == "__main__":
    main()
