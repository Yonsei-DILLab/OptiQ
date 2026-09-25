"""Train production Direct GMM OptiQ for 10k transitions on the four-way task."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import flax.serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from environment import FourWayEnv


class Replay:
    def __init__(self, capacity=100_000, seed=0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.obs = np.empty((capacity, 2), np.float32)
        self.actions = np.empty((capacity, 2), np.float32)
        self.next_obs = np.empty((capacity, 2), np.float32)
        self.rewards = np.empty((capacity, 1), np.float32)
        self.dones = np.empty((capacity, 1), np.float32)
        self.position = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.position
        self.obs[i], self.actions[i], self.next_obs[i] = obs, action, next_obs
        self.rewards[i, 0], self.dones[i, 0] = reward, done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, env=None):
        del env
        if self.size < batch_size:
            raise RuntimeError(f"replay has {self.size} rows, needs {batch_size}")
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        ids = self.rng.integers(0, self.size, size=batch_size)
        as_tensor = lambda x: torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)
        return ReplayBufferSamples(
            as_tensor(self.obs[ids]), as_tensor(self.actions[ids]),
            as_tensor(self.next_obs[ids]), as_tensor(self.dones[ids]),
            as_tensor(self.rewards[ids]),
        )


class SpaceOnlyEnv(gym.Env):
    """Supply spaces to the SB3-compatible learner; all steps use FourWayEnv."""
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Box(-7.0, 7.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("SpaceOnlyEnv is a descriptor; collect with FourWayEnv")

    def step(self, action):
        raise RuntimeError("SpaceOnlyEnv is a descriptor; collect with FourWayEnv")


def load_trainer():
    path = ROOT / "analysis_tools/experiments/20260921_gmm_trg_sweep/train.py"
    spec = importlib.util.spec_from_file_location("fourway_optiq_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_policy_config(trainer, out, budget, warmup, temperature):
    cfg = trainer.compose_config([
        "benchmark=ant", "seed=0", f"alg.actor.temperature={temperature}",
        "alg.actor.mean_output_init_scale=0.0001",
        "alg.actor.log_std_min=-5.0", "alg.actor.log_std_max=-1.0",
        "alg.actor.initial_log_std=-1.0",
        "alg.actor.teacher_std_floor=0.006737946999085467",
        "dacer.enabled=false", f"output_root={out}",
    ], allowed_policy_samples=64)
    cfg.total_steps = budget
    cfg.alg.learning_starts = warmup
    cfg.alg.actor.learning_starts = warmup
    cfg.alg.batch_size = 256
    cfg.alg.buffer_size = 100_000
    cfg.alg.gamma = 0.99
    cfg.alg.tau = 0.005
    cfg.alg.policy_tau = 1.0
    cfg.alg.policy_delay = 1
    cfg.alg.critic.n_atoms = 1
    cfg.alg.critic.backup_mode = "td"
    cfg.alg.critic.crossq_style = False
    cfg.alg.critic.hs = [256, 256]
    cfg.alg.actor.hidden_dims = [256, 256]
    cfg.alg.actor.include_anchor = False
    cfg.alg.actor.density_correction = True
    cfg.alg.actor.density_beta = 1.0
    cfg.alg.actor.adaptive_density_beta = False
    cfg.alg.actor.distillation_loss = "direct_gmm_nll"
    cfg.alg.actor.num_policy_samples = 64
    cfg.alg.actor.proposals_per_policy_sample = 1
    cfg.alg.actor.proposal_sampling_mode = "exact"
    cfg.alg.optimizer.lr_actor = 3e-4
    cfg.alg.optimizer.lr_critic = 3e-4
    cfg.wandb.activate = False
    return cfg


def sample_action(model, obs, key, conditional_noise=True):
    p = model.policy
    arr = np.asarray(p.sample_action(
        p.actor_state, jnp.asarray(obs[None], dtype=jnp.float32), key,
        deterministic=False, sample_conditional_noise=conditional_noise,
    ))
    return np.clip(arr[0], -1.0, 1.0).astype(np.float32)


def direction_id(action):
    if np.max(np.abs(action)) < 0.05:
        return -1
    axis = int(np.argmax(np.abs(action)))
    if axis == 0:
        return 0 if action[0] >= 0 else 1  # east, west
    return 2 if action[1] >= 0 else 3  # north, south


def directional_q(model):
    p = model.policy
    observations = jnp.zeros((4, 2), dtype=jnp.float32)
    actions = jnp.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=jnp.float32)
    prediction = p.qf_state.apply_fn(
        {"params": p.qf_state.params, "batch_stats": p.qf_state.batch_stats},
        observations, actions,
        rngs={"dropout": jax.random.PRNGKey(31415)}, train=False,
    )
    q = np.asarray(prediction)
    if q.ndim == 3 and q.shape[-1] == 1:
        q = q[..., 0]
    return q.astype(np.float64)


def evaluate(model, step, episodes, seed, out):
    eval_env = FourWayEnv()
    modes = {"policy": True, "mu_only": False}
    all_records = {}
    for mode, conditional_noise in modes.items():
        trajectories = np.full((episodes, eval_env.horizon + 1, 2), np.nan, np.float32)
        lengths = np.zeros(episodes, np.int32)
        returns = np.zeros(episodes, np.float32)
        success = np.zeros(episodes, np.int8)
        goals = np.full(episodes, -1, np.int8)
        first_actions = np.zeros((episodes, 2), np.float32)
        for ep in range(episodes):
            obs, _ = eval_env.reset(seed=seed + ep)
            trajectories[ep, 0] = obs
            key = jax.random.PRNGKey(seed * 1_000_003 + ep * 101 + step + (0 if mode == "policy" else 77))
            for t in range(eval_env.horizon):
                key, action_key = jax.random.split(key)
                action = sample_action(model, obs, action_key, conditional_noise)
                if t == 0:
                    first_actions[ep] = action
                obs, reward, terminated, truncated, info = eval_env.step(action)
                trajectories[ep, t + 1] = obs
                returns[ep] += reward
                lengths[ep] = t + 1
                if terminated:
                    success[ep], goals[ep] = 1, info["goal_id"]
                if terminated or truncated:
                    break
        directions = np.asarray([direction_id(a) for a in first_actions], np.int8)
        direction_counts = np.bincount(directions[directions >= 0], minlength=4)
        goal_counts = np.bincount(goals[goals >= 0], minlength=4)
        record = dict(
            step=step, mode=mode, episodes=episodes,
            success_rate=float(success.mean()), mean_return=float(returns.mean()),
            goal_counts=goal_counts.tolist(), direction_counts=direction_counts.tolist(),
            uncommitted_first_actions=int(np.sum(directions < 0)),
            mean_episode_length=float(lengths.mean()),
        )
        np.savez_compressed(
            out / f"eval_{step:05d}_{mode}.npz", trajectories=trajectories,
            lengths=lengths, returns=returns, success=success, goals=goals,
            first_actions=first_actions, goal_counts=goal_counts,
            direction_counts=direction_counts,
        )
        all_records[mode] = record
        eval_env.close()
    cloud_obs = jnp.zeros((2048, 2), dtype=jnp.float32)
    cloud_key = jax.random.PRNGKey(seed * 991 + step + 17)
    p = model.policy
    p.reset_noise()
    cloud_policy = np.asarray(p.sample_action(
        p.actor_state, cloud_obs, cloud_key, deterministic=False,
        sample_conditional_noise=True,
    ))
    cloud_mu = np.asarray(p.sample_action(
        p.actor_state, cloud_obs, jax.random.fold_in(cloud_key, 1),
        deterministic=False, sample_conditional_noise=False,
    ))
    np.savez_compressed(out / f"origin_action_cloud_{step:05d}.npz",
                        policy=cloud_policy, mu_only=cloud_mu,
                        directional_q=directional_q(model))
    return all_records


def scalar_metrics(model):
    out = {}
    for key, value in model.logger.name_to_value.items():
        if not any(name in key for name in (
            "actor_loss", "critic_loss", "current_q_values", "next_q_values",
            "source_ess_absolute", "source_ess_fraction", "source_q_std",
            "q_logit_std", "density_logit_std", "q_to_density_logit_std_ratio",
            "full_weighted_q_gain", "cross_critic_weighted_q_gain",
            "max_source_weight", "q_neglogq_correlation",
        )):
            continue
        try:
            arr = np.asarray(value)
            if arr.size == 1 and np.isfinite(arr).all():
                out[key] = float(arr.reshape(()))
        except (TypeError, ValueError):
            pass
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interim-episodes", type=int, default=40)
    parser.add_argument("--final-episodes", type=int, default=100)
    args = parser.parse_args()
    if args.steps <= args.warmup or args.warmup < 256:
        raise ValueError("steps must exceed warmup and warmup must fill batch 256")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "evaluations").mkdir()
    (args.out / "checkpoints").mkdir()

    trainer = load_trainer()
    cfg = load_policy_config(trainer, args.out, args.steps, args.warmup, args.temperature)
    from omegaconf import OmegaConf
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (args.out / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    commit = os.environ.get("OPTIQ_SOURCE_COMMIT")
    if not commit:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = dict(
        source_commit=commit, task="symmetric 2D four-way terminal goals",
        seed=args.seed, environment_transitions=args.steps, warmup_transitions=args.warmup,
        updates_per_transition=1, batch_size=256, evaluation_start="fixed origin",
        goals=FourWayEnv().goal_positions.tolist(), reward="-30*||a||^2 - min_g ||s_next-g||^2 + 10 on success",
        temperature=args.temperature, beta=1.0, num_policy_samples=64, proposals_per_policy_sample=1,
        actor_hidden_dims=[256, 256], critic_hidden_dims=[256, 256],
        mean_output_init_scale=1e-4, log_std_bounds=[-5.0, -1.0], initial_log_std=-1.0,
        gamma=0.99, critic_tau=0.005, dacer=False, noveld=False,
        evaluation=dict(interim_episodes=args.interim_episodes, final_episodes=args.final_episodes,
                        checkpoints=[0, 1000, 2000, 5000, args.steps], modes=["policy", "mu_only"]),
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    model = trainer.runner.OptiQDIME(
        "MlpPolicy", env=SpaceOnlyEnv(), cfg=cfg,
        model_save_path=None, save_every_n_steps=args.steps,
    )
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(str(args.out / "learner"), ["csv"]))
    replay = Replay(seed=args.seed)
    model.replay_buffer = replay
    model._total_timesteps = args.steps
    env = FourWayEnv()
    env.action_space.seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    checkpoints = {0, args.warmup, 2_000, 5_000, 10_000, args.steps}
    checkpoints = sorted(step for step in checkpoints if step <= args.steps)
    history = []
    eval_history = []
    eval_index = 0
    for step in range(1, args.steps + 1):
        if step <= args.warmup:
            action = env.action_space.sample().astype(np.float32)
        else:
            model.policy.reset_noise()
            action = sample_action(model, obs, model.policy.noise_key, conditional_noise=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = float(terminated or truncated)
        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()
        if step > args.warmup:
            model.num_timesteps = step
            model._current_progress_remaining = 1.0 - step / args.steps
            model.train(batch_size=256, gradient_steps=1)
            row = dict(step=step, learner_updates=model._n_updates, **scalar_metrics(model))
            history.append(row)
        if step in checkpoints:
            eval_episodes = args.final_episodes if step == args.steps else args.interim_episodes
            results = evaluate(model, step, eval_episodes, args.seed + 20_000, args.out / "evaluations")
            eval_history.extend(results.values())
            state = dict(actor=model.policy.actor_state, critic=model.policy.qf_state,
                         target_actor=model.policy.target_actor_state,
                         target_critic_params=model.policy.qf_state.target_params)
            (args.out / "checkpoints" / f"policy_{step:05d}.msgpack").write_bytes(
                flax.serialization.to_bytes(state))
            print(json.dumps(dict(step=step, learner_updates=model._n_updates,
                                  eval=results), sort_keys=True), flush=True)
    env.close()
    fields = sorted({k for row in history for k in row})
    with (args.out / "training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    (args.out / "evaluation_summary.json").write_text(json.dumps(eval_history, indent=2) + "\n")
    (args.out / "result.json").write_text(json.dumps(dict(
        source_commit=commit, status="complete", environment_transitions=args.steps,
        learner_updates=model._n_updates, evaluations=eval_history,
    ), indent=2) + "\n")
    print(json.dumps(dict(status="complete", steps=args.steps,
                          learner_updates=model._n_updates, out=str(args.out))), flush=True)


if __name__ == "__main__":
    main()
