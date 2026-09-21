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
                     q_from_output, sample_action)

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


def make_update(model, temperature, gamma, tau, huber_delta):
    @jax.jit
    def update(state, target_params, batch):
        target_output = model.apply({"params": target_params}, batch["next_obs"])
        target = jax.lax.stop_gradient(
            batch["rewards"] + gamma * batch["not_terminal"] * target_output["value"])
        def loss_fn(params):
            output = model.apply({"params": params}, batch["obs"])
            logp = log_prob_from_output(output, batch["actions"])
            q = output["value"] + temperature * logp
            loss = optax.huber_loss(q, target, delta=huber_delta).mean()
            return loss, (q, output["value"], logp)
        (loss, (q, value, logp)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        target_params = optax.incremental_update(state.params, target_params, tau)
        metrics = {"train/loss": loss, "train/q_mean": q.mean(),
                   "train/v_mean": value.mean(), "train/data_log_prob": logp.mean(),
                   "train/target_mean": target.mean(),
                   "train/abs_td": jnp.abs(q - target).mean(),
                   "train/grad_norm_preclip": grad_norm}
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
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--knots", type=int, default=33)
    parser.add_argument("--temperature", type=float, default=.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--tau", type=float, default=.005)
    parser.add_argument("--huber-delta", type=float, default=10.)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
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
    model = ConditionalSplineCircuit(action_dim=action_dim, rank=args.rank, knots=args.knots)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.asarray(obs, dtype=jnp.float32)[None])["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(10.), optax.adam(args.learning_rate))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    target_params = params
    update = make_update(model, args.temperature, args.gamma, args.tau, args.huber_delta)
    replay = ReplayBuffer(args.buffer_size, obs_dim, action_dim, args.seed + 101)
    source_root = Path(__file__).resolve().parents[3]
    command = sys.argv
    config = vars(args).copy(); config["output"] = str(config["output"])
    config.update(algorithm="spline_energy", algorithm_display_name="Spline Energy Circuit",
                  model_equation="Q=V+alpha*log_pi",
                  policy="rank-16 mixture of products of normalized positive linear splines",
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
    run = wandb.init(entity=os.getenv("WANDB_ENTITY", "gsmin2018"),
                     project=os.getenv("WANDB_PROJECT", "OptiQ-MuJoCo-Spline-Energy"),
                     group=os.getenv("WANDB_RUN_GROUP", "spline_energy"),
                     job_type="train" if args.total_steps == 1_000_000 else "smoke",
                     name=f"{args.env}-spline-energy-s{args.seed}", mode=args.wandb_mode,
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
                    path.write_bytes(serialization.to_bytes(
                        {"state": state, "target_params": target_params, "key": key, "env_steps": env_step}))
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
