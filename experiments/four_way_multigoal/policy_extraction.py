import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax


OUTPUT_DIR = Path("outputs/four_way_multigoal/policy_extraction")
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
MODE_CENTERS = np.asarray(
    [
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)
TARGET_WEIGHTS = np.full(4, 0.25, dtype=np.float32)
PARTICLE_COUNTS = (1, 2, 4, 16, 32, 64)
METHODS = ("ot_categorical", "independent_categorical")
SEEDS = (0, 1, 2)
TRAIN_STEPS = 3_000
CHUNK_STEPS = 500
SNAPSHOT_STEPS = (0, 500, 1_500, 3_000)
EVAL_SAMPLES = 20_000
NUM_SLICES = 256
ACTOR_WIDTH = 64
SINKHORN_EPSILON = 0.05
SINKHORN_ITERATIONS = 200
SINKHORN_BALANCING_ITERATIONS = 500
OPTIMIZER = optax.adam(3.0e-4)


def init_actor(key):
    keys = jax.random.split(key, 3)
    dimensions = (2, ACTOR_WIDTH, ACTOR_WIDTH, 2)
    params = []
    for layer_index, (key_i, fan_in, fan_out) in enumerate(
        zip(keys, dimensions[:-1], dimensions[1:])
    ):
        scale = np.sqrt(2.0 / fan_in) if layer_index < 2 else 0.01
        params.append(
            {
                "w": scale * jax.random.normal(key_i, (fan_in, fan_out)),
                "b": jnp.zeros((fan_out,)),
            }
        )
    return params


def actor_raw_apply(params, latent):
    hidden = latent
    for layer in params[:-1]:
        hidden = jax.nn.gelu(hidden @ layer["w"] + layer["b"])
    return hidden @ params[-1]["w"] + params[-1]["b"]


def actor_apply(params, latent):
    return jnp.clip(actor_raw_apply(params, latent), ACTION_LOW, ACTION_HIGH)


def sinkhorn_plan(actions):
    centers = jnp.asarray(MODE_CENTERS, dtype=actions.dtype)
    cost = jnp.sum(jnp.square(actions[:, None, :] - centers[None, :, :]), axis=-1)
    num_particles = actions.shape[0]
    log_kernel = -cost / jnp.asarray(SINKHORN_EPSILON, dtype=cost.dtype)
    log_row = jnp.full((num_particles,), -jnp.log(num_particles), dtype=cost.dtype)
    log_column = jnp.full((4,), -jnp.log(4.0), dtype=cost.dtype)

    def body(_, duals):
        log_u, log_v = duals
        log_u = log_row - jax.scipy.special.logsumexp(
            log_kernel + log_v[None, :], axis=1
        )
        log_v = log_column - jax.scipy.special.logsumexp(
            log_kernel + log_u[:, None], axis=0
        )
        return log_u, log_v

    initial = (jnp.zeros_like(log_row), jnp.zeros_like(log_column))
    log_u, log_v = jax.lax.fori_loop(
        0,
        SINKHORN_ITERATIONS,
        body,
        initial,
    )
    plan = jnp.maximum(
        jnp.exp(log_kernel + log_u[:, None] + log_v[None, :]),
        1.0e-30,
    )

    def balance(_, current_plan):
        current_plan = current_plan * (
            (1.0 / num_particles)
            / jnp.maximum(jnp.sum(current_plan, axis=1, keepdims=True), 1.0e-30)
        )
        current_plan = current_plan * (
            0.25
            / jnp.maximum(jnp.sum(current_plan, axis=0, keepdims=True), 1.0e-30)
        )
        return current_plan

    return jax.lax.fori_loop(
        0,
        SINKHORN_BALANCING_ITERATIONS,
        balance,
        plan,
    )


@partial(jax.jit, static_argnames=("num_particles", "use_ot"))
def train_chunk(params, optimizer_state, key, num_particles, use_ot):
    centers = jnp.asarray(MODE_CENTERS)

    def step(carry, _):
        current_params, current_optimizer_state, current_key = carry
        current_key, latent_key, target_key = jax.random.split(current_key, 3)
        latent = jax.random.normal(latent_key, (num_particles, 2))
        actions = actor_apply(current_params, latent)

        if use_ot:
            plan = sinkhorn_plan(actions)
            row_mass = jnp.sum(plan, axis=1, keepdims=True)
            conditionals = plan / jnp.maximum(row_mass, 1.0e-30)
            row_error = jnp.max(
                jnp.abs(jnp.sum(plan, axis=1) - 1.0 / num_particles)
            )
            column_error = jnp.max(
                jnp.abs(jnp.sum(plan, axis=0) - jnp.asarray(TARGET_WEIGHTS))
            )
            selection_marginal_error = jnp.max(
                jnp.abs(jnp.mean(conditionals, axis=0) - jnp.asarray(TARGET_WEIGHTS))
            )
        else:
            conditionals = jnp.full((num_particles, 4), 0.25)
            row_error = jnp.asarray(0.0)
            column_error = jnp.asarray(0.0)
            selection_marginal_error = jnp.asarray(0.0)

        target_indices = jax.random.categorical(
            target_key,
            jnp.log(jnp.maximum(conditionals, 1.0e-30)),
            axis=-1,
        )
        targets = centers[target_indices]

        def loss_fn(candidate_params):
            raw_actions = actor_raw_apply(candidate_params, latent)
            return jnp.mean(jnp.sum(jnp.square(raw_actions - targets), axis=-1))

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, next_optimizer_state = OPTIMIZER.update(
            gradients,
            current_optimizer_state,
            current_params,
        )
        next_params = optax.apply_updates(current_params, updates)
        metrics = jnp.asarray(
            [loss, row_error, column_error, selection_marginal_error]
        )
        return (next_params, next_optimizer_state, current_key), metrics

    (params, optimizer_state, key), metrics = jax.lax.scan(
        step,
        (params, optimizer_state, key),
        xs=None,
        length=CHUNK_STEPS,
    )
    return params, optimizer_state, key, metrics[-1], jnp.max(metrics[:, 1:], axis=0)


def sliced_wasserstein_2(actions, directions):
    actions = np.asarray(actions, dtype=np.float64)
    sample_count = len(actions)
    repeated_centers = np.repeat(MODE_CENTERS, sample_count // 4, axis=0)
    if len(repeated_centers) != sample_count:
        raise ValueError("Evaluation sample count must be divisible by four.")
    projected_actions = actions @ directions.T
    projected_target = repeated_centers @ directions.T
    projected_actions.sort(axis=0)
    projected_target.sort(axis=0)
    return float(np.sqrt(np.mean(np.square(projected_actions - projected_target))))


def distribution_metrics(actions, directions):
    actions = np.asarray(actions)
    distances = np.linalg.norm(
        actions[:, None, :] - MODE_CENTERS[None, :, :],
        axis=-1,
    )
    nearest = np.argmin(distances, axis=1)
    nearest_distance = np.min(distances, axis=1)
    assigned_mass = np.bincount(nearest, minlength=4) / len(actions)
    near_mode = nearest_distance <= 0.35
    near_mass = np.asarray(
        [np.mean(near_mode & (nearest == mode)) for mode in range(4)]
    )
    return {
        "sliced_w2": sliced_wasserstein_2(actions, directions),
        "nearest_mode_distance_mean": float(nearest_distance.mean()),
        "nearest_mode_distance_p90": float(np.quantile(nearest_distance, 0.9)),
        "nearest_assignment_mass": assigned_mass.tolist(),
        "near_mode_mass": near_mass.tolist(),
        "near_mode_total_mass": float(near_mass.sum()),
        "covered_modes": int(np.sum(near_mass >= 0.05)),
        "mode_mass_tv": float(0.5 * np.abs(assigned_mass - 0.25).sum()),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_covariance": np.cov(actions.T).tolist(),
    }


def train(seed, num_particles, method, train_steps, snapshot_steps):
    use_ot = method == "ot_categorical"
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_actor(init_key)
    optimizer_state = OPTIMIZER.init(params)
    eval_rng = np.random.default_rng(10_000 + seed)
    evaluation_latent = eval_rng.normal(size=(EVAL_SAMPLES, 2)).astype(np.float32)
    direction_rng = np.random.default_rng(20_000 + seed)
    directions = direction_rng.normal(size=(NUM_SLICES, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    snapshots = {}
    metrics = {}
    initial_actions = np.asarray(actor_apply(params, jnp.asarray(evaluation_latent)))
    snapshots[0] = initial_actions
    metrics[0] = distribution_metrics(initial_actions, directions)

    final_loss = np.nan
    max_row_error = 0.0
    max_column_error = 0.0
    max_selection_marginal_error = 0.0
    for completed_steps in range(CHUNK_STEPS, train_steps + 1, CHUNK_STEPS):
        params, optimizer_state, key, chunk_metrics, marginal_errors = train_chunk(
            params,
            optimizer_state,
            key,
            num_particles=num_particles,
            use_ot=use_ot,
        )
        final_loss = float(chunk_metrics[0])
        max_row_error = max(max_row_error, float(marginal_errors[0]))
        max_column_error = max(max_column_error, float(marginal_errors[1]))
        max_selection_marginal_error = max(
            max_selection_marginal_error,
            float(marginal_errors[2]),
        )
        if completed_steps in snapshot_steps:
            actions = np.asarray(actor_apply(params, jnp.asarray(evaluation_latent)))
            snapshots[completed_steps] = actions
            metrics[completed_steps] = distribution_metrics(actions, directions)

    return {
        "snapshots": snapshots,
        "metrics": metrics,
        "final_mse": final_loss,
        "max_sinkhorn_row_marginal_error": max_row_error,
        "max_sinkhorn_column_marginal_error": max_column_error,
        "max_selection_marginal_error": max_selection_marginal_error,
    }


def pooled_histogram(axis, actions, color, show_target=True):
    bins = np.linspace(-1.55, 1.55, 95)
    histogram, x_edges, y_edges = np.histogram2d(
        actions[:, 0],
        actions[:, 1],
        bins=(bins, bins),
        density=True,
    )
    masked = np.ma.masked_where(histogram.T <= 0.01, histogram.T)
    axis.imshow(
        masked,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap=color,
        interpolation="bilinear",
        aspect="equal",
        alpha=0.92,
    )
    if show_target:
        axis.scatter(
            MODE_CENTERS[:, 0],
            MODE_CENTERS[:, 1],
            marker="x",
            s=42,
            linewidth=1.7,
            color="black",
            zorder=5,
        )
    axis.axhline(0.0, color="black", linewidth=0.45, alpha=0.15)
    axis.axvline(0.0, color="black", linewidth=0.45, alpha=0.15)
    axis.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal")
    axis.grid(alpha=0.08)


def draw_ot_scaling(results, arrays, train_steps):
    display_steps = (0, 500, train_steps)
    particle_counts = results["setup"]["particle_counts_N"]
    figure, axes = plt.subplots(
        len(display_steps),
        len(particle_counts),
        figsize=(16.7, 8.4),
        sharex=True,
        sharey=True,
    )
    for row, step in enumerate(display_steps):
        for column, num_particles in enumerate(particle_counts):
            axis = axes[row, column]
            pooled = np.concatenate(
                [
                    arrays[f"ot_categorical_N{num_particles}_seed{seed}_step{step}"]
                    for seed in results["setup"]["seeds"]
                ],
                axis=0,
            )
            pooled_histogram(axis, pooled, "RdPu")
            values = [
                results["runs"]["ot_categorical"][str(num_particles)][str(seed)][
                    "metrics"
                ][str(step)]["sliced_w2"]
                for seed in results["setup"]["seeds"]
            ]
            axis.text(
                0.96,
                0.94,
                rf"$SW_2={np.mean(values):.3f}$",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            if row == 0:
                axis.set_title(rf"$N={num_particles}$", fontsize=11)
            if column == 0:
                axis.set_ylabel(f"Step {step:,}\nAction 2")
            if row == len(display_steps) - 1:
                axis.set_xlabel("Action 1")
    figure.suptitle(
        "Categorical OT distillation of a fixed four-way target",
        fontsize=14,
    )
    figure.tight_layout()
    path = OUTPUT_DIR / "fourway_ot_categorical_scaling.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def draw_method_comparison(results, arrays, train_steps):
    particle_counts = results["setup"]["particle_counts_N"]
    figure, axes = plt.subplots(
        len(METHODS),
        len(particle_counts),
        figsize=(16.7, 5.8),
        sharex=True,
        sharey=True,
    )
    labels = {
        "ot_categorical": "OT categorical",
        "independent_categorical": "Independent categorical",
    }
    colormaps = {
        "ot_categorical": "RdPu",
        "independent_categorical": "Blues",
    }
    for row, method in enumerate(METHODS):
        for column, num_particles in enumerate(particle_counts):
            axis = axes[row, column]
            pooled = np.concatenate(
                [
                    arrays[f"{method}_N{num_particles}_seed{seed}_step{train_steps}"]
                    for seed in results["setup"]["seeds"]
                ],
                axis=0,
            )
            pooled_histogram(axis, pooled, colormaps[method])
            values = [
                results["runs"][method][str(num_particles)][str(seed)]["metrics"][
                    str(train_steps)
                ]["sliced_w2"]
                for seed in results["setup"]["seeds"]
            ]
            axis.text(
                0.96,
                0.94,
                rf"$SW_2={np.mean(values):.3f}$",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            if row == 0:
                axis.set_title(rf"$N={num_particles}$", fontsize=11)
            if column == 0:
                axis.set_ylabel(f"{labels[method]}\nAction 2")
            if row == len(METHODS) - 1:
                axis.set_xlabel("Action 1")
    figure.suptitle(
        "Why transport correspondence matters for four-way distillation",
        fontsize=14,
    )
    figure.tight_layout()
    path = OUTPUT_DIR / "fourway_ot_vs_independent_final.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def draw_scaling_metrics(results, train_steps):
    particle_counts = np.asarray(results["setup"]["particle_counts_N"])
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    colors = {
        "ot_categorical": "#c51b7d",
        "independent_categorical": "#2166ac",
    }
    labels = {
        "ot_categorical": "OT categorical",
        "independent_categorical": "Independent categorical",
    }
    metric_specs = (
        ("sliced_w2", r"Final $SW_2$", None),
        ("near_mode_total_mass", "Mass within 0.35 of a mode", (0.0, 1.05)),
        ("covered_modes", "Covered modes", (-0.1, 4.1)),
    )
    for axis, (metric, ylabel, limits) in zip(axes, metric_specs):
        for method in METHODS:
            values = np.asarray(
                [
                    [
                        results["runs"][method][str(num_particles)][str(seed)][
                            "metrics"
                        ][str(train_steps)][metric]
                        for seed in results["setup"]["seeds"]
                    ]
                    for num_particles in particle_counts
                ]
            )
            axis.errorbar(
                particle_counts,
                values.mean(axis=1),
                yerr=values.std(axis=1),
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=colors[method],
                label=labels[method],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(particle_counts)
        axis.set_xticklabels(particle_counts)
        axis.set(xlabel="Policy samples per OT update N", ylabel=ylabel)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    path = OUTPUT_DIR / "fourway_particle_scaling_metrics.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def serializable_run(run):
    return {
        "metrics": {
            str(step): metrics for step, metrics in run["metrics"].items()
        },
        "final_mse": run["final_mse"],
        "max_sinkhorn_row_marginal_error": run[
            "max_sinkhorn_row_marginal_error"
        ],
        "max_sinkhorn_column_marginal_error": run[
            "max_sinkhorn_column_marginal_error"
        ],
        "max_selection_marginal_error": run["max_selection_marginal_error"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    global OUTPUT_DIR
    args = parse_args()
    output_dir = OUTPUT_DIR / "quick" if args.quick else OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    particle_counts = (1, 4, 16) if args.quick else PARTICLE_COUNTS
    seeds = (0,) if args.quick else SEEDS
    train_steps = 500 if args.quick else TRAIN_STEPS
    snapshot_steps = (0, 500) if args.quick else SNAPSHOT_STEPS
    arrays = {}
    results = {
        "setup": {
            "action_bounds": [ACTION_LOW, ACTION_HIGH],
            "target": "four equal Dirac modes at cardinal directions",
            "mode_centers": MODE_CENTERS.tolist(),
            "target_weights": TARGET_WEIGHTS.tolist(),
            "particle_counts_N": list(particle_counts),
            "methods": list(METHODS),
            "seeds": list(seeds),
            "train_steps": train_steps,
            "snapshot_steps": list(snapshot_steps),
            "actor": f"MLP {ACTOR_WIDTH}x{ACTOR_WIDTH} GELU",
            "latent": "N(0, I_2)",
            "sinkhorn_epsilon": SINKHORN_EPSILON,
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
            "sinkhorn_balancing_iterations": SINKHORN_BALANCING_ITERATIONS,
            "evaluation_metric": f"sliced W2 with {NUM_SLICES} projections",
        },
        "runs": {},
    }

    for method in METHODS:
        results["runs"][method] = {}
        for num_particles in particle_counts:
            results["runs"][method][str(num_particles)] = {}
            for seed in seeds:
                run = train(
                    seed,
                    num_particles,
                    method,
                    train_steps,
                    snapshot_steps,
                )
                for step, actions in run["snapshots"].items():
                    arrays[f"{method}_N{num_particles}_seed{seed}_step{step}"] = actions
                results["runs"][method][str(num_particles)][str(seed)] = (
                    serializable_run(run)
                )
                final_metrics = run["metrics"][train_steps]
                print(
                    f"finished method={method} N={num_particles} seed={seed} "
                    f"SW2={final_metrics['sliced_w2']:.5f} "
                    f"modes={final_metrics['covered_modes']}",
                    flush=True,
                )

    snapshot_path = OUTPUT_DIR / "snapshots.npz"
    np.savez_compressed(snapshot_path, **arrays)
    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    if not args.quick:
        ot_scaling_path = draw_ot_scaling(results, arrays, train_steps)
        comparison_path = draw_method_comparison(results, arrays, train_steps)
        metrics_path = draw_scaling_metrics(results, train_steps)
        print(
            json.dumps(
                {
                    "results": str(results_path),
                    "snapshots": str(snapshot_path),
                    "ot_scaling_figure": str(ot_scaling_path),
                    "comparison_figure": str(comparison_path),
                    "metrics_figure": str(metrics_path),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps({"results": str(results_path)}, indent=2))


if __name__ == "__main__":
    main()
