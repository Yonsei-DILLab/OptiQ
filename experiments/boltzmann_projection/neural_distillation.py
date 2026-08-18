import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.special import logsumexp


OUTPUT_DIR = Path("outputs/boltzmann_projection/neural_distillation")
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
TEMPERATURE = 1.0
NUM_PROPOSALS = 512
PARTICLE_COUNTS = (1, 2, 4, 16, 32, 64)
SEEDS = (0, 1, 2)
TRAIN_STEPS = 3_000
CHUNK_STEPS = 500
SNAPSHOT_STEPS = (0, 500, 1_500, 3_000)
EVAL_SAMPLES = 50_000
ACTOR_WIDTH = 64
OPTIMIZER = optax.adam(3.0e-4)


def q_value(actions):
    actions = np.asarray(actions)
    peaks = np.stack(
        [
            np.log(0.58) - 0.5 * ((actions + 0.85) / 0.22) ** 2,
            np.log(0.42) - 0.5 * ((actions - 0.75) / 0.34) ** 2,
        ],
        axis=0,
    )
    return logsumexp(peaks, axis=0)


def softmax(logits):
    logits = np.asarray(logits) - np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()


def make_target():
    width = (ACTION_HIGH - ACTION_LOW) / NUM_PROPOSALS
    proposals = ACTION_LOW + (np.arange(NUM_PROPOSALS) + 0.5) * width
    weights = softmax(q_value(proposals) / TEMPERATURE)
    return proposals.astype(np.float32), weights.astype(np.float32)


def monotone_row_conditionals(num_particles, target_weights):
    target_cdf = np.concatenate(([0.0], np.cumsum(target_weights)))
    target_cdf[-1] = 1.0
    conditionals = np.zeros((num_particles, len(target_weights)), dtype=np.float64)
    for row in range(num_particles):
        lower = row / num_particles
        upper = (row + 1) / num_particles
        overlaps = np.maximum(
            0.0,
            np.minimum(upper, target_cdf[1:])
            - np.maximum(lower, target_cdf[:-1]),
        )
        conditionals[row] = num_particles * overlaps
    conditionals /= conditionals.sum(axis=1, keepdims=True)
    return conditionals.astype(np.float32)


def init_actor(key):
    keys = jax.random.split(key, 3)
    dimensions = (1, ACTOR_WIDTH, ACTOR_WIDTH, 1)
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
    hidden = latent[:, None]
    for layer in params[:-1]:
        hidden = jax.nn.gelu(hidden @ layer["w"] + layer["b"])
    return (hidden @ params[-1]["w"] + params[-1]["b"])[:, 0]


def actor_apply(params, latent):
    return jnp.clip(actor_raw_apply(params, latent), ACTION_LOW, ACTION_HIGH)


@jax.jit
def train_chunk(params, optimizer_state, key, proposals, log_conditionals):
    num_particles = log_conditionals.shape[0]

    def step(carry, _):
        current_params, current_optimizer_state, current_key = carry
        current_key, latent_key, target_key = jax.random.split(current_key, 3)
        latent = jax.random.normal(latent_key, (num_particles,))
        actions = actor_apply(current_params, latent)
        order = jnp.argsort(actions)
        target_indices_by_rank = jax.random.categorical(
            target_key,
            log_conditionals,
            axis=-1,
        )
        targets_by_rank = proposals[target_indices_by_rank]
        targets = jnp.zeros_like(targets_by_rank).at[order].set(targets_by_rank)

        def loss_fn(candidate_params):
            raw_actions = actor_raw_apply(candidate_params, latent)
            return jnp.mean(jnp.square(raw_actions - targets))

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, next_optimizer_state = OPTIMIZER.update(
            gradients,
            current_optimizer_state,
            current_params,
        )
        next_params = optax.apply_updates(current_params, updates)
        return (next_params, next_optimizer_state, current_key), loss

    (params, optimizer_state, key), losses = jax.lax.scan(
        step,
        (params, optimizer_state, key),
        xs=None,
        length=CHUNK_STEPS,
    )
    return params, optimizer_state, key, losses[-1]


def weighted_wasserstein_2(samples_a, weights_a, samples_b, weights_b):
    order_a = np.argsort(samples_a)
    order_b = np.argsort(samples_b)
    samples_a = np.asarray(samples_a, dtype=np.float64)[order_a]
    samples_b = np.asarray(samples_b, dtype=np.float64)[order_b]
    weights_a = np.asarray(weights_a, dtype=np.float64)[order_a].copy()
    weights_b = np.asarray(weights_b, dtype=np.float64)[order_b].copy()
    weights_a /= weights_a.sum()
    weights_b /= weights_b.sum()

    i = 0
    j = 0
    remaining_a = weights_a[0]
    remaining_b = weights_b[0]
    squared_cost = 0.0
    tolerance = 1.0e-15
    while i < len(samples_a) and j < len(samples_b):
        transported = min(remaining_a, remaining_b)
        squared_cost += transported * (samples_a[i] - samples_b[j]) ** 2
        remaining_a -= transported
        remaining_b -= transported
        if remaining_a <= tolerance:
            i += 1
            if i < len(samples_a):
                remaining_a = weights_a[i]
        if remaining_b <= tolerance:
            j += 1
            if j < len(samples_b):
                remaining_b = weights_b[j]
    return float(np.sqrt(max(squared_cost, 0.0)))


def evaluate_actor(params, evaluation_latent, proposals, target_weights):
    actions = np.asarray(actor_apply(params, jnp.asarray(evaluation_latent)))
    w2 = weighted_wasserstein_2(
        actions,
        np.full(len(actions), 1.0 / len(actions)),
        proposals,
        target_weights,
    )
    return actions, w2


def train(seed, num_particles, proposals, target_weights):
    conditionals = monotone_row_conditionals(num_particles, target_weights)
    log_conditionals = np.log(np.maximum(conditionals, 1.0e-30)).astype(np.float32)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_actor(init_key)
    optimizer_state = OPTIMIZER.init(params)
    evaluation_latent = np.random.default_rng(10_000 + seed).normal(
        size=EVAL_SAMPLES
    ).astype(np.float32)

    snapshots = {}
    snapshot_w2 = {}
    actions, w2 = evaluate_actor(
        params,
        evaluation_latent,
        proposals,
        target_weights,
    )
    snapshots[0] = actions
    snapshot_w2[0] = w2

    final_loss = np.nan
    for completed_steps in range(CHUNK_STEPS, TRAIN_STEPS + 1, CHUNK_STEPS):
        params, optimizer_state, key, final_loss = train_chunk(
            params,
            optimizer_state,
            key,
            jnp.asarray(proposals),
            jnp.asarray(log_conditionals),
        )
        if completed_steps in SNAPSHOT_STEPS:
            actions, w2 = evaluate_actor(
                params,
                evaluation_latent,
                proposals,
                target_weights,
            )
            snapshots[completed_steps] = actions
            snapshot_w2[completed_steps] = w2

    return {
        "snapshots": {str(step): actions.tolist() for step, actions in snapshots.items()},
        "snapshot_w2": {str(step): value for step, value in snapshot_w2.items()},
        "final_mse": float(final_loss),
        "row_conditional_marginal_error": float(
            np.max(
                np.abs(
                    conditionals.mean(axis=0)
                    - np.asarray(target_weights)
                )
            )
        ),
    }


def draw_distribution_grid(results, proposals, target_weights):
    particle_counts = results["setup"]["particle_counts_N"]
    display_steps = (0, 500, 3_000)
    colors = plt.cm.magma(np.linspace(0.35, 0.78, len(display_steps)))
    width = (ACTION_HIGH - ACTION_LOW) / NUM_PROPOSALS
    target_density = target_weights / width
    target_edges = np.linspace(ACTION_LOW, ACTION_HIGH, NUM_PROPOSALS + 1)
    histogram_edges = np.linspace(ACTION_LOW, ACTION_HIGH, 121)

    figure, axes = plt.subplots(
        len(display_steps),
        len(particle_counts),
        figsize=(17.0, 8.0),
        sharex=True,
        sharey=True,
    )
    for row, (step, color) in enumerate(zip(display_steps, colors)):
        for column, num_particles in enumerate(particle_counts):
            axis = axes[row, column]
            runs = results["runs"][str(num_particles)]
            pooled = np.concatenate(
                [np.asarray(run["snapshots"][str(step)]) for run in runs]
            )
            axis.stairs(
                target_density,
                target_edges,
                color="black",
                linewidth=1.5,
                label="Soft-Q target" if row == 0 and column == 0 else None,
            )
            axis.hist(
                pooled,
                bins=histogram_edges,
                density=True,
                histtype="stepfilled",
                color=color,
                alpha=0.48,
                label="Categorical actor" if row == 0 and column == 0 else None,
            )
            values = np.asarray(
                [run["snapshot_w2"][str(step)] for run in runs]
            )
            axis.text(
                0.97,
                0.92,
                rf"$W_2={values.mean():.3f}$",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            if row == 0:
                axis.set_title(rf"$N={num_particles}$", fontsize=11)
            if column == 0:
                axis.set_ylabel(f"Step {step:,}\nDensity")
            if row == len(display_steps) - 1:
                axis.set_xlabel("Action")
            axis.grid(alpha=0.12)

    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    figure.suptitle(
        "Categorical OT distillation with a fixed bimodal soft-Q target (M=512)",
        fontsize=14,
    )
    figure.tight_layout()
    path = OUTPUT_DIR / "categorical_nn_scaling_distributions.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def draw_final_w2(results):
    particle_counts = results["setup"]["particle_counts_N"]
    means = []
    standard_deviations = []
    for num_particles in particle_counts:
        values = np.asarray(
            [
                run["snapshot_w2"][str(TRAIN_STEPS)]
                for run in results["runs"][str(num_particles)]
            ]
        )
        means.append(values.mean())
        standard_deviations.append(values.std())

    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.errorbar(
        particle_counts,
        means,
        yerr=standard_deviations,
        marker="o",
        capsize=4,
        color="#c51b7d",
        linewidth=2.0,
    )
    axis.set_xscale("log", base=2)
    axis.set(
        title="Neural categorical projection vs. particles per update",
        xlabel="Categorical OT particles N",
        ylabel=r"Final $W_2(\pi_\theta,\rho_M^Q)$",
        xticks=particle_counts,
    )
    axis.set_xticklabels(particle_counts)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = OUTPUT_DIR / "categorical_nn_scaling_w2.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    proposals, target_weights = make_target()
    results = {
        "setup": {
            "action_bounds": [ACTION_LOW, ACTION_HIGH],
            "temperature": TEMPERATURE,
            "proposal_count_M": NUM_PROPOSALS,
            "particle_counts_N": list(PARTICLE_COUNTS),
            "seeds": list(SEEDS),
            "train_steps": TRAIN_STEPS,
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "actor": f"MLP {ACTOR_WIDTH}x{ACTOR_WIDTH} GELU",
            "assignment": "categorical exact 1D monotone OT",
        },
        "runs": {},
    }

    for num_particles in PARTICLE_COUNTS:
        runs = []
        for seed in SEEDS:
            result = train(seed, num_particles, proposals, target_weights)
            runs.append(result)
            print(
                f"finished N={num_particles} seed={seed} "
                f"W2={result['snapshot_w2'][str(TRAIN_STEPS)]:.6f}"
            )
        results["runs"][str(num_particles)] = runs

    distribution_path = draw_distribution_grid(results, proposals, target_weights)
    w2_path = draw_final_w2(results)
    results["distribution_figure"] = str(distribution_path)
    results["w2_figure"] = str(w2_path)
    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({
        "results": str(results_path),
        "distribution_figure": str(distribution_path),
        "w2_figure": str(w2_path),
    }, indent=2))


if __name__ == "__main__":
    main()
