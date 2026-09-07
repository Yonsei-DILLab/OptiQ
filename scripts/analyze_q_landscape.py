#!/usr/bin/env python3
"""Compare local action-space Q landscapes across the four DMC Dog tasks.

The diagnostic deliberately uses the same physical states and latent noises when
``--probe shared`` is selected.  For every actor action it aligns the horizontal
slice axis with the gradient of the twin-mean Q, so averaging slices does not
erase a real high-Q direction.  It also evaluates the actual 38-dimensional
truncated-Gaussian proposal cloud used by OptiQ; the 2-D plot alone can otherwise
make a high-dimensional critic look artificially flat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from hydra import compose, initialize_config_dir

from diffusion.dime import load_state
from optiq_dime.policy import OptiQPolicy
from optiq_dime.critic_utils import critic_expectation
from optiq_dime.transport import sample_truncated_gaussian


TASKS = ("run", "trot", "walk", "stand")
COLORS = {
    "run": "#4477AA",
    "trot": "#EE6677",
    "walk": "#228833",
    "stand": "#AA3377",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probe",
        choices=("shared", "own", *TASKS),
        default="shared",
        help=(
            "shared: random-policy states; own: each task's own replay probe; "
            "a task name: use that task's replay probe for every critic"
        ),
    )
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--latents-per-state", type=int, default=4)
    parser.add_argument("--proposals", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument(
        "--slice-radius",
        type=float,
        default=0.45,
        help="L2 action displacement.  0.45 is close to a typical 38-D sigma=0.1 proposal.",
    )
    parser.add_argument("--proposal-std", type=float, default=0.1)
    parser.add_argument("--proposal-clip", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def compose_cfg(task: str):
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        version_base=None, config_dir=str(repo_root / "configs")
    ):
        return compose(
            config_name="optiq_dime_dog",
            overrides=[
                f"task={task}",
                "seed=1",
                "wandb.activate=false",
                "checkpoint_interval=0",
            ],
        )


def find_checkpoint_dir(root: Path, task: str, step: int) -> Path:
    matches = sorted(root.glob(f"*_{task}_*"))
    matches = [
        path
        for path in matches
        if (path / f"actor_state_{step}.msgpack").exists()
        and (path / f"critic_state_{step}.msgpack").exists()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {task} checkpoint at step {step}, found {matches}"
        )
    return matches[0]


def build_policy(task: str, checkpoint_dir: Path, step: int):
    cfg = compose_cfg(task)
    env = gym.make(cfg.env_name)
    policy = OptiQPolicy(env.observation_space, env.action_space, cfg)
    policy.build(
        jax.random.PRNGKey(int(cfg.seed)),
        lambda _: float(cfg.alg.optimizer.lr_actor),
        float(cfg.alg.optimizer.lr_critic),
    )
    policy.actor_state = load_state(
        str(checkpoint_dir), "actor_state", step, train_state=policy.actor_state
    )
    policy.qf_state = load_state(
        str(checkpoint_dir), "critic_state", step, train_state=policy.qf_state
    )
    return cfg, env, policy


def flatten_observation(observation_space, observation) -> np.ndarray:
    return np.asarray(
        spaces.flatten(observation_space, observation), dtype=np.float32
    ).reshape(-1)


def collect_shared_probe(
    count: int, seed: int, burn_in: int = 100, stride: int = 7
) -> np.ndarray:
    """Collect task-independent physical Dog states using random actions."""
    env = gym.make("dm_control/dog-stand")
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    result = []
    total = burn_in + count * stride
    for index in range(total):
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(
            np.float32
        )
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset(seed=seed + index + 1)
        if index >= burn_in and (index - burn_in) % stride == 0:
            result.append(flatten_observation(env.observation_space, observation))
    env.close()
    return np.stack(result[:count])


def load_own_probe(checkpoint_dir: Path, count: int) -> np.ndarray:
    path = checkpoint_dir / "landscape_probe_batch.npz"
    if not path.exists():
        raise FileNotFoundError(f"own probe does not exist yet: {path}")
    with np.load(path) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
    if observations.shape[0] < count:
        raise RuntimeError(f"probe has {observations.shape[0]} states, need {count}")
    return observations[:count]


def pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    return np.divide(
        (x * y).sum(axis=1),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1.0e-12,
    )


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def analyze_task(
    task: str,
    checkpoint_dir: Path,
    step: int,
    observations: np.ndarray,
    latents: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg, env, policy = build_policy(task, checkpoint_dir, step)
    n_states, n_latents, action_dim = latents.shape
    repeated_obs = np.repeat(observations[:, None, :], n_latents, axis=1).reshape(
        n_states * n_latents, -1
    )
    flat_latents = latents.reshape(n_states * n_latents, action_dim)
    anchors = policy.actor_state.apply_fn(
        {"params": policy.actor_state.params},
        jnp.asarray(repeated_obs),
        jnp.asarray(flat_latents),
    )
    anchors = np.asarray(jnp.clip(anchors, -1.0, 1.0), dtype=np.float32)
    z_atoms = jnp.linspace(
        float(cfg.alg.critic.v_min),
        float(cfg.alg.critic.v_max),
        int(cfg.alg.critic.n_atoms),
    )

    @jax.jit
    def evaluate_q(obs, actions):
        distributions = policy.qf_state.apply_fn(
            {
                "params": policy.qf_state.params,
                "batch_stats": policy.qf_state.batch_stats,
            },
            obs,
            actions,
            rngs={"dropout": jax.random.PRNGKey(0)},
            train=False,
        )
        return critic_expectation(distributions, z_atoms)

    def q_mean_one(action, observation):
        twins = evaluate_q(observation[None], action[None])[:, 0]
        return jnp.mean(twins)

    gradient_fn = jax.jit(jax.vmap(jax.grad(q_mean_one), in_axes=(0, 0)))
    gradients = np.asarray(
        gradient_fn(jnp.asarray(anchors), jnp.asarray(repeated_obs))
    )
    gradient_norms = np.linalg.norm(gradients, axis=1)
    direction_1 = gradients / np.maximum(gradient_norms[:, None], 1.0e-10)

    # A deterministic orthogonal direction exposes curvature without changing the
    # horizontal high-Q direction chosen above.
    rng = np.random.default_rng(args.seed + TASKS.index(task) * 1009)
    direction_2 = rng.normal(size=direction_1.shape).astype(np.float32)
    direction_2 -= (
        np.sum(direction_2 * direction_1, axis=1, keepdims=True) * direction_1
    )
    direction_2 /= np.maximum(
        np.linalg.norm(direction_2, axis=1, keepdims=True), 1.0e-10
    )

    # Actual high-dimensional OptiQ proposal cloud.
    proposals = sample_truncated_gaussian(
        jax.random.PRNGKey(args.seed + TASKS.index(task) * 17),
        jnp.asarray(anchors),
        repeats=args.proposals,
        std=args.proposal_std,
        perturb_clip=args.proposal_clip,
        include_anchor=False,
    )
    proposal_obs = np.repeat(repeated_obs[:, None, :], args.proposals, axis=1)
    proposal_twins = np.asarray(
        evaluate_q(
            jnp.asarray(proposal_obs.reshape(-1, repeated_obs.shape[-1])),
            proposals.reshape(-1, action_dim),
        )
    ).reshape(2, anchors.shape[0], args.proposals)
    anchor_twins = np.asarray(
        evaluate_q(jnp.asarray(repeated_obs), jnp.asarray(anchors))
    )
    proposal_mean = proposal_twins.mean(axis=0)
    proposal_min = proposal_twins.min(axis=0)
    anchor_mean = anchor_twins.mean(axis=0)
    anchor_min = anchor_twins.min(axis=0)
    proposal_delta = proposal_mean - anchor_mean[:, None]
    proposal_min_delta = proposal_min - anchor_min[:, None]

    # Gradient-aligned 2-D slice.
    coordinates = np.linspace(
        -args.slice_radius, args.slice_radius, args.grid_size, dtype=np.float32
    )
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    dx = xx.reshape(1, -1, 1)
    dy = yy.reshape(1, -1, 1)
    grid_actions = (
        anchors[:, None, :]
        + dx * direction_1[:, None, :]
        + dy * direction_2[:, None, :]
    )
    grid_actions = np.clip(grid_actions, -1.0, 1.0)
    grid_obs = np.repeat(repeated_obs[:, None, :], xx.size, axis=1)
    grid_twins = np.asarray(
        evaluate_q(
            jnp.asarray(grid_obs.reshape(-1, repeated_obs.shape[-1])),
            jnp.asarray(grid_actions.reshape(-1, action_dim)),
        )
    ).reshape(2, anchors.shape[0], args.grid_size, args.grid_size)
    grid_twin_delta = grid_twins - anchor_twins[:, :, None, None]
    grid_mean_delta = grid_twins.mean(axis=0) - anchor_mean[:, None, None]
    grid_min_delta = grid_twins.min(axis=0) - anchor_min[:, None, None]

    tail_count = max(1, int(round(0.05 * args.proposals)))
    twin_top_overlap = []
    for first, second in zip(proposal_twins[0], proposal_twins[1]):
        top_first = set(np.argpartition(first, -tail_count)[-tail_count:].tolist())
        top_second = set(np.argpartition(second, -tail_count)[-tail_count:].tolist())
        twin_top_overlap.append(
            len(top_first & top_second) / max(1, len(top_first | top_second))
        )

    per_anchor_max = proposal_delta.max(axis=1)
    positive_halfmax_fraction = np.mean(
        proposal_delta
        >= np.maximum(0.0, 0.5 * per_anchor_max)[:, None],
        axis=1,
    )
    metrics = {
        "task": task,
        "step": step,
        "probe": args.probe,
        "anchors": int(anchors.shape[0]),
        "action_dim": int(action_dim),
        "anchor_q_mean": summarize(anchor_mean),
        "anchor_q_min": summarize(anchor_min),
        "gradient_l2_norm": summarize(gradient_norms),
        "proposal_delta_max": summarize(per_anchor_max),
        "proposal_delta_p99": summarize(np.quantile(proposal_delta, 0.99, axis=1)),
        "proposal_delta_p95": summarize(np.quantile(proposal_delta, 0.95, axis=1)),
        "proposal_delta_range": summarize(np.ptp(proposal_delta, axis=1)),
        "proposal_min_delta_max": summarize(proposal_min_delta.max(axis=1)),
        "proposal_improvement_fraction": summarize(
            np.mean(proposal_delta > 0.0, axis=1)
        ),
        "proposal_high_region_halfmax_fraction": summarize(
            positive_halfmax_fraction
        ),
        "proposal_fraction_delta_gt_0p005": summarize(
            np.mean(proposal_delta > 0.005, axis=1)
        ),
        "proposal_fraction_delta_gt_0p01": summarize(
            np.mean(proposal_delta > 0.01, axis=1)
        ),
        "proposal_fraction_delta_gt_0p02": summarize(
            np.mean(proposal_delta > 0.02, axis=1)
        ),
        "twin_proposal_correlation": summarize(
            pearson_rows(proposal_twins[0], proposal_twins[1])
        ),
        "twin_top5pct_jaccard": summarize(np.asarray(twin_top_overlap)),
        "grid_delta_range": summarize(
            np.ptp(grid_mean_delta.reshape(anchors.shape[0], -1), axis=1)
        ),
        "grid_min_delta_range": summarize(
            np.ptp(grid_min_delta.reshape(anchors.shape[0], -1), axis=1)
        ),
    }
    result = {
        "metrics": metrics,
        "coordinates": coordinates,
        "grid_mean_delta": grid_mean_delta,
        "grid_min_delta": grid_min_delta,
        "grid_twin_delta": grid_twin_delta,
        "proposal_delta": proposal_delta,
        "proposal_min_delta": proposal_min_delta,
    }
    env.close()
    del policy
    return result


def plot_surfaces(results: dict[str, dict[str, Any]], args: argparse.Namespace):
    task_surfaces = {
        task: np.median(results[task]["grid_mean_delta"], axis=0) for task in TASKS
    }
    all_values = np.concatenate([surface.ravel() for surface in task_surfaces.values()])
    common_limit = float(np.quantile(np.abs(all_values), 0.995))
    common_limit = max(common_limit, 1.0e-6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    image = None
    for axis, task in zip(axes, TASKS):
        coordinates = results[task]["coordinates"]
        image = axis.imshow(
            task_surfaces[task],
            origin="lower",
            extent=[coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]],
            cmap="coolwarm",
            vmin=-common_limit,
            vmax=common_limit,
            aspect="equal",
        )
        metrics = results[task]["metrics"]
        axis.set_title(
            f"{task.capitalize()}\n"
            f"median max ΔQ={metrics['proposal_delta_max']['median']:.4f}"
        )
        axis.set_xlabel("L2 displacement along ∇a Q")
        if task == "run":
            axis.set_ylabel("orthogonal L2 displacement")
        axis.scatter([0.0], [0.0], marker="x", color="black", s=25, linewidths=1)
    fig.colorbar(image, ax=axes, label="median twin-mean ΔQ (common scale)")
    fig.suptitle(
        f"Gradient-aligned local Q landscape at step {args.step} ({args.probe} states)"
    )
    path = args.output_dir / f"q_landscape_step{args.step}_{args.probe}_surfaces.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_proposals(results: dict[str, dict[str, Any]], args: argparse.Namespace):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    quantiles = np.linspace(0.0, 1.0, 201)
    for task in TASKS:
        delta = results[task]["proposal_delta"].reshape(-1)
        axes[0].plot(
            quantiles,
            np.quantile(delta, quantiles),
            color=COLORS[task],
            label=task,
            linewidth=2,
        )
        per_anchor_max = results[task]["proposal_delta"].max(axis=1)
        axes[1].plot(
            np.sort(per_anchor_max),
            np.linspace(0.0, 1.0, per_anchor_max.size),
            color=COLORS[task],
            label=task,
            linewidth=2,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("proposal ΔQ quantile")
    axes[0].set_ylabel("twin-mean Q(proposal) − Q(anchor)")
    axes[0].set_title("All 38-D truncated-Gaussian proposals")
    axes[0].grid(alpha=0.2)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("best ΔQ among proposals for each anchor")
    axes[1].set_ylabel("fraction of anchors ≤ x")
    axes[1].set_title("Discoverable high-Q tail")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle(
        f"Q variation in the actual proposal cloud at step {args.step} ({args.probe} states)"
    )
    path = args.output_dir / f"q_landscape_step{args.step}_{args.probe}_proposals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_twin_surfaces(results: dict[str, dict[str, Any]], args: argparse.Namespace):
    surfaces = {
        (task, twin): np.median(
            results[task]["grid_twin_delta"][twin], axis=0
        )
        for task in TASKS
        for twin in range(2)
    }
    all_values = np.concatenate([surface.ravel() for surface in surfaces.values()])
    common_limit = max(float(np.quantile(np.abs(all_values), 0.995)), 1.0e-6)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
    image = None
    for column, task in enumerate(TASKS):
        coordinates = results[task]["coordinates"]
        for twin in range(2):
            axis = axes[twin, column]
            image = axis.imshow(
                surfaces[(task, twin)],
                origin="lower",
                extent=[
                    coordinates[0],
                    coordinates[-1],
                    coordinates[0],
                    coordinates[-1],
                ],
                cmap="coolwarm",
                vmin=-common_limit,
                vmax=common_limit,
                aspect="equal",
            )
            axis.scatter(
                [0.0], [0.0], marker="x", color="black", s=22, linewidths=1
            )
            if twin == 0:
                axis.set_title(task.capitalize())
            if column == 0:
                axis.set_ylabel(f"Q{twin + 1}: orthogonal displacement")
            if twin == 1:
                axis.set_xlabel("displacement along mean-Q gradient")
    fig.colorbar(image, ax=axes, label="median ΔQ (common scale)")
    fig.suptitle(
        f"Twin critic landscapes at step {args.step} ({args.probe} states)"
    )
    path = args.output_dir / f"q_landscape_step{args.step}_{args.probe}_twins.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_dim = 38
    rng = np.random.default_rng(args.seed)
    latents = rng.normal(
        size=(args.states, args.latents_per_state, action_dim)
    ).astype(np.float32)
    if args.probe == "shared":
        shared_probe = collect_shared_probe(args.states, args.seed)
    elif args.probe in TASKS:
        source_checkpoint_dir = find_checkpoint_dir(
            args.checkpoint_root, args.probe, args.step
        )
        shared_probe = load_own_probe(source_checkpoint_dir, args.states)
    else:
        shared_probe = None

    results = {}
    for task in TASKS:
        checkpoint_dir = find_checkpoint_dir(
            args.checkpoint_root, task, args.step
        )
        observations = (
            shared_probe
            if shared_probe is not None
            else load_own_probe(checkpoint_dir, args.states)
        )
        print(f"analyzing {task} at step {args.step} ...", flush=True)
        results[task] = analyze_task(
            task,
            checkpoint_dir,
            args.step,
            observations,
            latents,
            args,
        )
        print(json.dumps(results[task]["metrics"], indent=2), flush=True)

    surface_path = plot_surfaces(results, args)
    proposal_path = plot_proposals(results, args)
    twin_path = plot_twin_surfaces(results, args)
    metrics_path = (
        args.output_dir / f"q_landscape_step{args.step}_{args.probe}_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(
            {task: results[task]["metrics"] for task in TASKS}, indent=2
        )
        + "\n"
    )
    npz_path = args.output_dir / f"q_landscape_step{args.step}_{args.probe}_raw.npz"
    np.savez_compressed(
        npz_path,
        **{
            f"{task}_{name}": value
            for task, result in results.items()
            for name, value in result.items()
            if name != "metrics"
        },
    )
    print(f"saved {surface_path}")
    print(f"saved {proposal_path}")
    print(f"saved {twin_path}")
    print(f"saved {metrics_path}")
    print(f"saved {npz_path}")


if __name__ == "__main__":
    main()
