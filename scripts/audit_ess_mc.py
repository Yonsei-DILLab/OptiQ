#!/usr/bin/env python3
"""Monte-Carlo audit of OptiQ importance weights at a fixed checkpoint.

The audit reproduces the exact 16 anchors x 5 candidates used by an actor
update, but repeatedly redraws actor latents and truncated-Gaussian noise.  It
separates Q-only, density-only, and combined weights and decomposes ESS into
competition between anchor groups and degeneracy within each group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_q_landscape import TASKS, build_policy, find_checkpoint_dir, load_own_probe
from optiq_dime.transport import sample_truncated_gaussian, truncated_mixture_log_density
from optiq_dime.critic_utils import critic_expectation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--proposal-std", type=float, default=0.1)
    parser.add_argument("--proposal-clip", type=float, default=0.15)
    return parser.parse_args()


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def row_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - np.mean(x, axis=-1, keepdims=True)
    y = y - np.mean(y, axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))
    return np.divide(
        np.sum(x * y, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1.0e-12,
    )


def ess(weights: np.ndarray) -> np.ndarray:
    return 1.0 / np.sum(np.square(weights), axis=-1)


def weighted_gain(weights: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(weights * q, axis=-1) - np.mean(q, axis=-1)


def scalar_mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def replicate_metrics(
    twin_q: np.ndarray,
    density_score: np.ndarray,
    radii: np.ndarray,
    temperature: float,
) -> dict[str, float]:
    """Return state-averaged metrics for one independent proposal redraw."""
    mean_q = np.mean(twin_q, axis=0)
    states = mean_q.shape[0]
    flat_q = mean_q.reshape(states, 80)
    flat_q1 = twin_q[0].reshape(states, 80)
    flat_q2 = twin_q[1].reshape(states, 80)
    flat_density = density_score.reshape(states, 80)
    flat_radii = radii.reshape(states, 80)
    q_score = flat_q / temperature

    q_weights = softmax(q_score)
    density_weights = softmax(flat_density)
    full_weights = softmax(q_score + flat_density)
    q1_weights = softmax(flat_q1 / temperature + flat_density)
    q2_weights = softmax(flat_q2 / temperature + flat_density)

    sorted_weights = np.sort(full_weights, axis=-1)[:, ::-1]
    full_groups = full_weights.reshape(states, 16, 5)
    q_groups = q_weights.reshape(states, 16, 5)
    density_groups = density_weights.reshape(states, 16, 5)
    density_scores_grouped = flat_density.reshape(states, 16, 5)
    group_mass = np.sum(full_groups, axis=-1)
    conditional = np.divide(
        full_groups,
        group_mass[..., None],
        out=np.full_like(full_groups, 0.2),
        where=group_mass[..., None] > 1.0e-20,
    )
    conditional_ess_fraction = ess(conditional) / 5.0
    weighted_conditional_ess = np.sum(
        group_mass * conditional_ess_fraction, axis=-1
    )

    top_index = np.argmax(full_weights, axis=-1)
    top_radius = np.take_along_axis(
        flat_radii, top_index[:, None], axis=-1
    )[:, 0]
    density_top_index = np.argmax(density_weights, axis=-1)
    cross_gain = 0.5 * (
        weighted_gain(q1_weights, flat_q2) + weighted_gain(q2_weights, flat_q1)
    )

    return {
        "full_ess_fraction": scalar_mean(ess(full_weights) / 80.0),
        "full_ess_absolute": scalar_mean(ess(full_weights)),
        "q_only_ess_fraction": scalar_mean(ess(q_weights) / 80.0),
        "density_only_ess_fraction": scalar_mean(ess(density_weights) / 80.0),
        "anchor_group_ess_fraction": scalar_mean(ess(group_mass) / 16.0),
        "within_anchor_ess_fraction_unweighted": scalar_mean(
            conditional_ess_fraction
        ),
        "within_anchor_ess_fraction_mass_weighted": scalar_mean(
            weighted_conditional_ess
        ),
        "top1_weight_mass": scalar_mean(sorted_weights[:, 0]),
        "top3_weight_mass": scalar_mean(np.sum(sorted_weights[:, :3], axis=-1)),
        "top5_weight_mass": scalar_mean(np.sum(sorted_weights[:, :5], axis=-1)),
        "q_logit_std": scalar_mean(np.std(q_score, axis=-1)),
        "density_logit_std": scalar_mean(np.std(flat_density, axis=-1)),
        "density_logit_std_perturbations_only": scalar_mean(
            np.std(density_scores_grouped[..., 1:].reshape(states, 64), axis=-1)
        ),
        "density_perturbation_minus_anchor_score": scalar_mean(
            np.mean(density_scores_grouped[..., 1:], axis=(1, 2))
            - np.mean(density_scores_grouped[..., 0], axis=1)
        ),
        "combined_logit_std": scalar_mean(
            np.std(q_score + flat_density, axis=-1)
        ),
        "q_to_density_logit_std_ratio": scalar_mean(
            np.std(q_score, axis=-1)
            / np.maximum(np.std(flat_density, axis=-1), 1.0e-12)
        ),
        "q_density_score_correlation": scalar_mean(
            row_correlation(q_score, flat_density)
        ),
        "density_radius_correlation": scalar_mean(
            row_correlation(flat_density, flat_radii)
        ),
        "full_density_weight_tv": scalar_mean(
            0.5 * np.sum(np.abs(full_weights - density_weights), axis=-1)
        ),
        "q_changes_density_argmax_fraction": scalar_mean(
            top_index != density_top_index
        ),
        "full_anchor_weight_mass": scalar_mean(
            np.sum(full_groups[..., 0], axis=-1)
        ),
        "density_only_anchor_weight_mass": scalar_mean(
            np.sum(density_groups[..., 0], axis=-1)
        ),
        "q_only_anchor_weight_mass": scalar_mean(
            np.sum(q_groups[..., 0], axis=-1)
        ),
        "top_weight_is_anchor_fraction": scalar_mean(top_index % 5 == 0),
        "uniform_candidate_radius": scalar_mean(flat_radii),
        "full_weighted_candidate_radius": scalar_mean(
            np.sum(full_weights * flat_radii, axis=-1)
        ),
        "top_weight_candidate_radius": scalar_mean(top_radius),
        "local_q_range": scalar_mean(np.ptp(mean_q, axis=-1)),
        "q_only_weighted_q_gain": scalar_mean(weighted_gain(q_weights, flat_q)),
        "density_only_weighted_q_gain": scalar_mean(
            weighted_gain(density_weights, flat_q)
        ),
        "full_weighted_q_gain": scalar_mean(weighted_gain(full_weights, flat_q)),
        "cross_twin_full_weighted_q_gain": scalar_mean(cross_gain),
    }


def aggregate(replicates: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for metric in replicates[0]:
        values = np.asarray([item[metric] for item in replicates])
        result[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        if metric.endswith("gain"):
            result[metric]["positive_fraction"] = float(np.mean(values > 0.0))
    return result


def plot_results(results: dict, output_path: Path, step: int) -> None:
    task_labels = [task.capitalize() for task in TASKS]
    colors = ["#4477AA", "#EE6677", "#228833", "#AA3377"]
    x = np.arange(len(TASKS))

    def means(metric):
        return np.asarray([results[t]["summary"][metric]["mean"] for t in TASKS])

    def errors(metric):
        mean = means(metric)
        low = np.asarray(
            [results[t]["summary"][metric]["ci95_low"] for t in TASKS]
        )
        high = np.asarray(
            [results[t]["summary"][metric]["ci95_high"] for t in TASKS]
        )
        return np.vstack((mean - low, high - mean))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    width = 0.25
    for offset, metric, label in (
        (-width, "full_ess_fraction", "full"),
        (0.0, "density_only_ess_fraction", "density only"),
        (width, "q_only_ess_fraction", "Q only"),
    ):
        axes[0, 0].bar(x + offset, means(metric), width, label=label)
        axes[0, 0].errorbar(
            x + offset, means(metric), yerr=errors(metric), fmt="none", color="black"
        )
    axes[0, 0].set_title("Global ESS fraction (80 candidates)")
    axes[0, 0].set_xticks(x, task_labels)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend()

    for offset, metric, label in (
        (-width / 2, "anchor_group_ess_fraction", "between 16 anchors"),
        (width / 2, "within_anchor_ess_fraction_mass_weighted", "within 5 candidates"),
    ):
        axes[0, 1].bar(x + offset, means(metric), width, label=label)
        axes[0, 1].errorbar(
            x + offset, means(metric), yerr=errors(metric), fmt="none", color="black"
        )
    axes[0, 1].set_title("Where full-weight degeneracy occurs")
    axes[0, 1].set_xticks(x, task_labels)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()

    for task, color in zip(TASKS, colors):
        axes[1, 0].scatter(
            means("density_logit_std")[TASKS.index(task)],
            means("q_logit_std")[TASKS.index(task)],
            s=90,
            color=color,
            label=task.capitalize(),
        )
    axes[1, 0].set_xlabel("std(-log q_F)")
    axes[1, 0].set_ylabel("std(Q / tau)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Logit-scale competition")
    axes[1, 0].legend()

    radius_metrics = (
        ("uniform_candidate_radius", "uniform candidates"),
        ("full_weighted_candidate_radius", "full-weighted"),
        ("top_weight_candidate_radius", "top-weight candidate"),
    )
    width = 0.23
    for index, (metric, label) in enumerate(radius_metrics):
        offset = (index - 1) * width
        axes[1, 1].bar(x + offset, means(metric), width, label=label)
        axes[1, 1].errorbar(
            x + offset, means(metric), yerr=errors(metric), fmt="none", color="black"
        )
    axes[1, 1].set_title("38-D distance from generating anchor")
    axes[1, 1].set_xticks(x, task_labels)
    axes[1, 1].set_ylabel("L2 radius")
    axes[1, 1].legend()

    fig.suptitle(
        f"Monte-Carlo ESS audit at step {step}: fixed critic, redrawn actor/noise"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for task_index, task in enumerate(TASKS):
        checkpoint = find_checkpoint_dir(args.checkpoint_root, task, args.step)
        observations = load_own_probe(checkpoint, args.states)
        cfg, env, policy = build_policy(task, checkpoint, args.step)
        action_dim = int(env.action_space.shape[0])
        z_atoms = jnp.linspace(
            float(cfg.alg.critic.v_min),
            float(cfg.alg.critic.v_max),
            int(cfg.alg.critic.n_atoms),
        )

        @jax.jit
        def actor_apply(obs, latent):
            return jnp.clip(
                policy.actor_state.apply_fn(
                    {"params": policy.actor_state.params}, obs, latent
                ),
                -1.0,
                1.0,
            )

        @jax.jit
        def critic_apply(obs, actions):
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

        candidate_observations = np.repeat(
            observations[:, None, :], 80, axis=1
        ).reshape(args.states * 80, -1)
        task_replicates = []
        for replicate in range(args.replicates):
            replicate_seed = args.seed + 100003 * task_index + replicate
            rng = np.random.default_rng(replicate_seed)
            latents = rng.normal(
                size=(args.states, 16, action_dim)
            ).astype(np.float32)
            repeated_observations = np.repeat(
                observations[:, None, :], 16, axis=1
            ).reshape(args.states * 16, -1)
            anchors = actor_apply(
                jnp.asarray(repeated_observations),
                jnp.asarray(latents.reshape(args.states * 16, action_dim)),
            )
            candidates = sample_truncated_gaussian(
                jax.random.PRNGKey(replicate_seed),
                anchors,
                repeats=5,
                std=args.proposal_std,
                perturb_clip=args.proposal_clip,
                include_anchor=True,
            ).reshape(args.states, 16, 5, action_dim)
            anchor_grid = anchors.reshape(args.states, 16, action_dim)
            density_score = -np.asarray(
                truncated_mixture_log_density(
                    candidates.reshape(args.states, 80, action_dim),
                    anchor_grid,
                    args.proposal_std,
                    args.proposal_clip,
                )
            )
            twin_q = np.asarray(
                critic_apply(
                    jnp.asarray(candidate_observations),
                    candidates.reshape(args.states * 80, action_dim),
                )
            ).reshape(2, args.states, 16, 5)
            radii = np.linalg.norm(
                np.asarray(candidates) - np.asarray(anchor_grid)[:, :, None, :],
                axis=-1,
            )
            metrics = replicate_metrics(
                twin_q, density_score, radii, args.temperature
            )
            task_replicates.append(metrics)
            print(
                f"{task} {replicate + 1:02d}/{args.replicates}: "
                f"ESS={metrics['full_ess_absolute']:.2f}/80, "
                f"group={metrics['anchor_group_ess_fraction']:.3f}, "
                f"within={metrics['within_anchor_ess_fraction_mass_weighted']:.3f}",
                flush=True,
            )

        results[task] = {
            "summary": aggregate(task_replicates),
            "replicates": task_replicates,
        }
        env.close()

    results["metadata"] = {
        "step": args.step,
        "states": args.states,
        "replicates": args.replicates,
        "temperature": args.temperature,
        "proposal_std": args.proposal_std,
        "proposal_clip": args.proposal_clip,
        "seed": args.seed,
    }
    json_path = args.output_dir / f"ess_mc_audit_step{args.step}.json"
    png_path = args.output_dir / f"ess_mc_audit_step{args.step}.png"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    plot_results(results, png_path, args.step)
    print(f"saved {json_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
