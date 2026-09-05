#!/usr/bin/env python3
"""Controlled audit of whether a DMC Dog critic has useful local Q structure.

Unlike ``analyze_q_landscape.py``, this script never chooses visualization axes
from Q.  Each row fixes one task's replay states, actor anchors, latent samples,
and truncated-Gaussian perturbations, then feeds those exact same ``(s, a)``
pairs to all four critics.  The candidate layout exactly matches one OptiQ actor
update: 16 actor anchors and 5 candidates per anchor (anchor + 4 perturbations).
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

from analyze_q_landscape import (
    TASKS,
    build_policy,
    find_checkpoint_dir,
    load_own_probe,
)
from optiq_dime.proposals import (
    PROPOSAL_FAMILIES,
    proposal_log_density,
    q_action_gradients,
    sample_proposals,
    stabilize_proposal_log_density,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--proposal-std", type=float, default=0.1)
    parser.add_argument("--proposal-clip", type=float, default=0.15)
    parser.add_argument(
        "--proposal-family",
        choices=PROPOSAL_FAMILIES,
        default="isotropic_truncated",
    )
    parser.add_argument(
        "--proposal-sampling-mode",
        choices=("stratified", "exact"),
        default="stratified",
    )
    parser.add_argument("--proposal-perpendicular-std-ratio", type=float, default=0.5)
    parser.add_argument("--gamma-shape", type=float, default=2.0)
    parser.add_argument("--gamma-scale", type=float, default=0.1)
    parser.add_argument("--gamma-perpendicular-std", type=float, default=0.1)
    parser.add_argument(
        "--clip-untruncated-proposals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def row_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    return np.divide(
        np.sum(x * y, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1.0e-12,
    )


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "median", "p10", "p90")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def grouped_local_metrics(
    mean_q: np.ndarray,
    component_indices: np.ndarray,
    anchor_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure local Q statistics by the component that generated each sample."""
    states, num_components, repeats = mean_q.shape
    flat_q = mean_q.reshape(states, num_components * repeats)
    flat_components = component_indices.reshape(states, num_components * repeats)
    flat_anchor_mask = anchor_mask.reshape(states, num_components * repeats)
    anchor_q = mean_q[..., 0]
    local_range = np.full((states, num_components), np.nan, dtype=mean_q.dtype)
    local_best_gain = np.full_like(local_range, np.nan)
    local_improvement = np.full_like(local_range, np.nan)

    for state in range(states):
        for component in range(num_components):
            component_mask = flat_components[state] == component
            values = flat_q[state, component_mask]
            if values.size == 0:
                continue
            local_range[state, component] = np.ptp(values)
            local_best_gain[state, component] = (
                np.max(values) - anchor_q[state, component]
            )
            random_values = flat_q[
                state, component_mask & ~flat_anchor_mask[state]
            ]
            if random_values.size > 0:
                local_improvement[state, component] = np.mean(
                    random_values > anchor_q[state, component]
                )
    return local_range, local_best_gain, local_improvement


def candidate_metrics(
    twin_q: np.ndarray,
    density_score: np.ndarray,
    component_indices: np.ndarray,
    anchor_mask: np.ndarray,
    temperature: float,
) -> dict[str, dict[str, float]]:
    """Metrics over [state, 16 anchors, 5 candidates]."""
    mean_q = twin_q.mean(axis=0)
    states = mean_q.shape[0]
    flat_q = mean_q.reshape(states, -1)
    flat_q1 = twin_q[0].reshape(states, -1)
    flat_q2 = twin_q[1].reshape(states, -1)
    local_range, local_best_gain, local_improvement = grouped_local_metrics(
        mean_q, component_indices, anchor_mask
    )

    q_score = flat_q / temperature
    density_score = density_score.reshape(states, -1)
    q_weights = softmax(q_score)
    density_weights = softmax(density_score)
    full_weights = softmax(q_score + density_score)

    def weighted_gain(weights, q):
        return np.sum(weights * q, axis=-1) - np.mean(q, axis=-1)

    q1_weights = softmax(flat_q1 / temperature + density_score)
    q2_weights = softmax(flat_q2 / temperature + density_score)
    cross_twin_gain = 0.5 * (
        weighted_gain(q1_weights, flat_q2)
        + weighted_gain(q2_weights, flat_q1)
    )
    full_ess = 1.0 / np.sum(full_weights * full_weights, axis=-1)
    return {
        "anchor_q": summarize(mean_q[..., 0].reshape(-1)),
        "local_q_range": summarize(local_range),
        "local_best_q_gain": summarize(local_best_gain),
        "local_improvement_fraction": summarize(local_improvement),
        "q_logit_std": summarize(np.std(q_score, axis=-1)),
        "twin_candidate_correlation": summarize(
            row_correlation(flat_q1, flat_q2)
        ),
        "q_only_weighted_q_gain": summarize(weighted_gain(q_weights, flat_q)),
        "density_only_weighted_q_gain": summarize(
            weighted_gain(density_weights, flat_q)
        ),
        "full_weighted_q_gain": summarize(weighted_gain(full_weights, flat_q)),
        "cross_twin_full_weighted_q_gain": summarize(cross_twin_gain),
        "full_ess_fraction": summarize(full_ess / flat_q.shape[-1]),
    }


def plot_matrix(results, output_path: Path, step: int, sampling_mode: str):
    specifications = (
        ("local_q_range", "mean local Q range", "magma"),
        ("q_logit_std", "mean std(Q / tau)", "viridis"),
        ("full_weighted_q_gain", "mean full-weight Q gain", "plasma"),
        (
            "cross_twin_full_weighted_q_gain",
            "mean cross-twin Q gain",
            "cividis",
        ),
    )
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    for axis, (metric, title, cmap) in zip(axes, specifications):
        values = np.asarray(
            [
                [results[reference][critic][metric]["mean"] for critic in TASKS]
                for reference in TASKS
            ]
        )
        image = axis.imshow(values, cmap=cmap, aspect="equal")
        axis.set_xticks(range(4), [task.capitalize() for task in TASKS])
        axis.set_yticks(range(4), [task.capitalize() for task in TASKS])
        axis.set_xlabel("critic")
        axis.set_ylabel("fixed state + actor + noise source")
        axis.set_title(title)
        midpoint = 0.5 * (values.min() + values.max())
        for row in range(4):
            for column in range(4):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    color="white" if value > midpoint else "black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, shrink=0.72)
    fig.suptitle(
        f"Controlled Q-region audit at step {step} ({sampling_mode}): "
        "identical (s, a) within each row"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        task: find_checkpoint_dir(args.checkpoint_root, task, args.step)
        for task in TASKS
    }
    loaded = {
        task: build_policy(task, checkpoints[task], args.step) for task in TASKS
    }
    action_dim = 38
    rng = np.random.default_rng(args.seed)
    common_latents = rng.normal(
        size=(args.states, 16, action_dim)
    ).astype(np.float32)
    results = {}
    out_of_bounds_by_reference = {}
    unsupported_by_reference = {}

    for reference_index, reference in enumerate(TASKS):
        print(f"building fixed candidates from {reference}", flush=True)
        observations = load_own_probe(checkpoints[reference], args.states)
        reference_cfg, _, reference_policy = loaded[reference]
        repeated_observations = np.repeat(observations[:, None, :], 16, axis=1)
        anchors = reference_policy.actor_state.apply_fn(
            {"params": reference_policy.actor_state.params},
            jnp.asarray(repeated_observations.reshape(args.states * 16, -1)),
            jnp.asarray(common_latents.reshape(args.states * 16, action_dim)),
        )
        anchors = jnp.clip(anchors, -1.0, 1.0)
        anchor_grid = anchors.reshape(args.states, 16, action_dim)
        q_gradients = None
        if args.proposal_family != "isotropic_truncated":
            reference_z_atoms = jnp.linspace(
                float(reference_cfg.alg.critic.v_min),
                float(reference_cfg.alg.critic.v_max),
                int(reference_cfg.alg.critic.n_atoms),
            )
            q_gradients = q_action_gradients(
                reference_policy.qf_state,
                jnp.asarray(observations),
                anchor_grid,
                reference_z_atoms,
                jax.random.PRNGKey(args.seed + reference_index * 97 + 1),
                "mean",
            )
        candidates, component_indices, out_of_bounds_fraction = sample_proposals(
            jax.random.PRNGKey(args.seed + reference_index * 97),
            anchor_grid,
            q_gradients,
            5,
            args.proposal_sampling_mode,
            args.proposal_family,
            args.proposal_std,
            args.proposal_clip,
            True,
            args.proposal_perpendicular_std_ratio,
            args.gamma_shape,
            args.gamma_scale,
            args.gamma_perpendicular_std,
            args.clip_untruncated_proposals,
        )
        out_of_bounds_by_reference[reference] = float(out_of_bounds_fraction)
        candidates = candidates.reshape(args.states, 16, 5, action_dim)
        component_indices = np.asarray(component_indices)
        anchor_mask = np.zeros((args.states, 16, 5), dtype=bool)
        anchor_mask[..., 0] = True
        flat_candidates = candidates.reshape(args.states, 80, action_dim)
        log_density = proposal_log_density(
            candidates.reshape(args.states, 80, action_dim),
            anchor_grid,
            q_gradients,
            args.proposal_family,
            args.proposal_std,
            args.proposal_clip,
            args.proposal_perpendicular_std_ratio,
            args.gamma_shape,
            args.gamma_scale,
            args.gamma_perpendicular_std,
        )
        unsupported_by_reference[reference] = float(
            jnp.mean(~jnp.isfinite(log_density))
        )
        density_score = -np.asarray(stabilize_proposal_log_density(log_density))
        candidate_observations = np.repeat(
            observations[:, None, :], 80, axis=1
        ).reshape(args.states * 80, -1)
        results[reference] = {}

        for critic in TASKS:
            cfg, _, policy = loaded[critic]
            z_atoms = jnp.linspace(
                float(cfg.alg.critic.v_min),
                float(cfg.alg.critic.v_max),
                int(cfg.alg.critic.n_atoms),
            )
            distributions = policy.qf_state.apply_fn(
                {
                    "params": policy.qf_state.params,
                    "batch_stats": policy.qf_state.batch_stats,
                },
                jnp.asarray(candidate_observations),
                candidates.reshape(args.states * 80, action_dim),
                rngs={"dropout": jax.random.PRNGKey(0)},
                train=False,
            )
            twin_q = np.asarray(
                jnp.sum(distributions * z_atoms, axis=-1)
            ).reshape(2, args.states, 16, 5)
            results[reference][critic] = candidate_metrics(
                twin_q,
                density_score,
                component_indices,
                anchor_mask,
                args.temperature,
            )
            diagonal = " [ON-POLICY]" if critic == reference else ""
            print(
                f"  {reference} actions -> {critic} critic{diagonal}: "
                f"range={results[reference][critic]['local_q_range']['mean']:.6f}, "
                f"full_gain={results[reference][critic]['full_weighted_q_gain']['mean']:.6f}, "
                f"cross_gain={results[reference][critic]['cross_twin_full_weighted_q_gain']['mean']:.6f}",
                flush=True,
            )

    results["metadata"] = {
        "step": args.step,
        "states": args.states,
        "seed": args.seed,
        "temperature": args.temperature,
        "proposal_std": args.proposal_std,
        "proposal_clip": args.proposal_clip,
        "proposal_family": args.proposal_family,
        "proposal_sampling_mode": args.proposal_sampling_mode,
        "proposal_perpendicular_std_ratio": args.proposal_perpendicular_std_ratio,
        "gamma_shape": args.gamma_shape,
        "gamma_scale": args.gamma_scale,
        "gamma_perpendicular_std": args.gamma_perpendicular_std,
        "clip_untruncated_proposals": args.clip_untruncated_proposals,
        "proposal_out_of_bounds_fraction": out_of_bounds_by_reference,
        "proposal_unsupported_fraction": unsupported_by_reference,
    }
    mode_tag = f"{args.proposal_family}_{args.proposal_sampling_mode}"
    metrics_path = args.output_dir / f"q_region_audit_{mode_tag}_step{args.step}.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    figure_path = args.output_dir / f"q_region_audit_{mode_tag}_step{args.step}.png"
    plot_matrix(
        results,
        figure_path,
        args.step,
        f"{args.proposal_family}/{args.proposal_sampling_mode}",
    )
    print(f"saved {metrics_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
