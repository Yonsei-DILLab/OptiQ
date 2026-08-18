import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.special import logsumexp, ndtri


OUTPUT_DIR = Path("outputs/boltzmann_projection/sample_refinement")
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
TEMPERATURE = 1.0
PROPOSAL_COUNTS = (8, 32, 128, 512)
POLICY_SAMPLE_COUNTS = (1, 2, 4, 16, 32, 64)
SEEDS = (0, 1, 2)
TRAIN_STEPS = 4_000
EVAL_SAMPLES = 50_000
REFERENCE_PROPOSALS = 131_072
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


def make_target(num_proposals):
    bin_width = (ACTION_HIGH - ACTION_LOW) / num_proposals
    proposals = ACTION_LOW + (np.arange(num_proposals) + 0.5) * bin_width
    weights = softmax(q_value(proposals) / TEMPERATURE)
    return proposals, weights


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


def optimal_equal_mass_policy_samples(proposals, target_weights, num_policy_samples):
    # In one dimension, optimal quadratic transport is the monotone quantile
    # coupling. Each policy sample represents one interval of mass 1 / N.
    cumulative = np.concatenate(([0.0], np.cumsum(target_weights)))
    cumulative[-1] = 1.0
    policy_samples = np.empty(num_policy_samples, dtype=np.float64)

    for sample_index in range(num_policy_samples):
        lower = sample_index / num_policy_samples
        upper = (sample_index + 1) / num_policy_samples
        first_moment = 0.0
        for proposal_index, proposal in enumerate(proposals):
            overlap = max(
                0.0,
                min(upper, cumulative[proposal_index + 1])
                - max(lower, cumulative[proposal_index]),
            )
            first_moment += overlap * proposal
        policy_samples[sample_index] = num_policy_samples * first_moment

    sample_weights = np.full(num_policy_samples, 1.0 / num_policy_samples)
    error = weighted_wasserstein_2(
        policy_samples,
        sample_weights,
        proposals,
        target_weights,
    )
    return policy_samples, error


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
def train_actor(params, optimizer_state, latent, targets):
    def step(carry, _):
        current_params, current_optimizer_state = carry

        def loss_fn(candidate_params):
            predicted = actor_raw_apply(candidate_params, latent)
            return jnp.mean(jnp.square(predicted - targets))

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, next_optimizer_state = OPTIMIZER.update(
            gradients,
            current_optimizer_state,
            current_params,
        )
        next_params = optax.apply_updates(current_params, updates)
        return (next_params, next_optimizer_state), loss

    (params, optimizer_state), losses = jax.lax.scan(
        step,
        (params, optimizer_state),
        xs=None,
        length=TRAIN_STEPS,
    )
    return params, optimizer_state, losses[-1]


def latent_anchors(num_policy_samples):
    quantiles = (np.arange(num_policy_samples) + 0.5) / num_policy_samples
    return ndtri(quantiles).astype(np.float32)


def run_neural_realization(seed, target_samples, proposals, target_weights):
    num_policy_samples = len(target_samples)
    anchors = latent_anchors(num_policy_samples)
    params = init_actor(jax.random.PRNGKey(seed))
    optimizer_state = OPTIMIZER.init(params)
    params, _, final_loss = train_actor(
        params,
        optimizer_state,
        jnp.asarray(anchors),
        jnp.asarray(target_samples, dtype=jnp.float32),
    )

    realized_slots = np.asarray(actor_apply(params, jnp.asarray(anchors)))
    slot_weights = np.full(num_policy_samples, 1.0 / num_policy_samples)
    slot_w2 = weighted_wasserstein_2(
        realized_slots,
        slot_weights,
        proposals,
        target_weights,
    )

    evaluation_latent = np.random.default_rng(100_000 + seed).normal(
        size=EVAL_SAMPLES
    ).astype(np.float32)
    actor_samples = np.asarray(
        actor_apply(params, jnp.asarray(evaluation_latent))
    )
    actor_w2 = weighted_wasserstein_2(
        actor_samples,
        np.full(EVAL_SAMPLES, 1.0 / EVAL_SAMPLES),
        proposals,
        target_weights,
    )
    return {
        "final_mse": float(final_loss),
        "slot_w2": slot_w2,
        "full_actor_w2": actor_w2,
        "realized_slots": realized_slots.tolist(),
    }


def verify_integer_refinement(errors):
    violations = []
    for smaller in POLICY_SAMPLE_COUNTS:
        for larger in POLICY_SAMPLE_COUNTS:
            if larger > smaller and larger % smaller == 0:
                if errors[larger] > errors[smaller] + 1.0e-11:
                    violations.append(
                        {
                            "N": smaller,
                            "kN": larger,
                            "W2_N": errors[smaller],
                            "W2_kN": errors[larger],
                        }
                    )
    return violations


def proposal_colors():
    palette = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4")
    return dict(zip(PROPOSAL_COUNTS, palette))


def plot_results(results):
    colors = proposal_colors()
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    proposals, weights = make_target(512)
    bin_width = (ACTION_HIGH - ACTION_LOW) / 512
    axes[0, 0].stairs(
        weights / bin_width,
        np.linspace(ACTION_LOW, ACTION_HIGH, 513),
        color="black",
        linewidth=2.0,
    )
    rug_counts = (1, 2, 4, 16, 64)
    rug_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rug_counts)))
    density_max = float(np.max(weights / bin_width))
    for row, (num_policy_samples, color) in enumerate(zip(rug_counts, rug_colors)):
        policy_samples = np.asarray(
            results["exact"]["512"][str(num_policy_samples)]["policy_samples"]
        )
        level = -0.07 * density_max * (row + 1)
        axes[0, 0].scatter(
            policy_samples,
            np.full_like(policy_samples, level),
            marker="|",
            s=90,
            linewidths=1.5,
            color=color,
            label=f"N={num_policy_samples}",
            clip_on=False,
        )
    axes[0, 0].set_ylim(-0.42 * density_max, 1.08 * density_max)
    axes[0, 0].set(
        title="Optimal equal-mass samples refine the bimodal target",
        xlabel="Action",
        ylabel="Soft-Q density",
    )
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    for num_proposals in PROPOSAL_COUNTS:
        exact_errors = [
            results["exact"][str(num_proposals)][str(n)]["w2"]
            for n in POLICY_SAMPLE_COUNTS
        ]
        axes[0, 1].plot(
            POLICY_SAMPLE_COUNTS,
            exact_errors,
            marker="o",
            color=colors[num_proposals],
            label=f"M={num_proposals}",
        )
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(
        title="Exact Wasserstein projection",
        xlabel="Policy samples N",
        ylabel=r"$W_2(\pi^*_{N,M},\rho_M^Q)$",
        xticks=POLICY_SAMPLE_COUNTS,
    )
    axes[0, 1].set_xticklabels(POLICY_SAMPLE_COUNTS)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    for num_proposals in PROPOSAL_COUNTS:
        means = []
        standard_deviations = []
        for num_policy_samples in POLICY_SAMPLE_COUNTS:
            runs = results["neural"][str(num_proposals)][str(num_policy_samples)]
            values = np.asarray([run["slot_w2"] for run in runs])
            means.append(values.mean())
            standard_deviations.append(values.std())
        axes[1, 0].errorbar(
            POLICY_SAMPLE_COUNTS,
            means,
            yerr=standard_deviations,
            marker="o",
            capsize=3,
            color=colors[num_proposals],
            label=f"M={num_proposals}",
        )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        title="NN realization at the fixed latent slots",
        xlabel="Distilled latent slots N",
        ylabel=r"$W_2(\hat\pi_{\theta,N},\rho_M^Q)$",
        xticks=POLICY_SAMPLE_COUNTS,
    )
    axes[1, 0].set_xticklabels(POLICY_SAMPLE_COUNTS)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False)

    num_proposals = 512
    exact = np.asarray(
        [
            results["exact"][str(num_proposals)][str(n)]["w2"]
            for n in POLICY_SAMPLE_COUNTS
        ]
    )
    slot_means = []
    slot_stds = []
    actor_means = []
    actor_stds = []
    for num_policy_samples in POLICY_SAMPLE_COUNTS:
        runs = results["neural"][str(num_proposals)][str(num_policy_samples)]
        slots = np.asarray([run["slot_w2"] for run in runs])
        actors = np.asarray([run["full_actor_w2"] for run in runs])
        slot_means.append(slots.mean())
        slot_stds.append(slots.std())
        actor_means.append(actors.mean())
        actor_stds.append(actors.std())
    axes[1, 1].plot(
        POLICY_SAMPLE_COUNTS,
        exact,
        marker="o",
        color="black",
        label="Ideal samples",
    )
    axes[1, 1].errorbar(
        POLICY_SAMPLE_COUNTS,
        slot_means,
        yerr=slot_stds,
        marker="s",
        capsize=3,
        color="#d62728",
        label="NN at anchors",
    )
    axes[1, 1].errorbar(
        POLICY_SAMPLE_COUNTS,
        actor_means,
        yerr=actor_stds,
        marker="^",
        capsize=3,
        color="#1f77b4",
        label="Full latent actor",
    )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        title="Ideal refinement vs. neural amortization (M=512)",
        xlabel="Policy samples N",
        ylabel=r"$W_2$ to fixed soft-Q target",
        xticks=POLICY_SAMPLE_COUNTS,
    )
    axes[1, 1].set_xticklabels(POLICY_SAMPLE_COUNTS)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    figure.tight_layout()
    figure_path = OUTPUT_DIR / "fixed_proposal_sample_refinement.png"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    main_figure_path = plot_exact_main(results)
    return figure_path, main_figure_path


def plot_exact_main(results):
    colors = proposal_colors()
    proposals, weights = make_target(512)
    del proposals
    bin_width = (ACTION_HIGH - ACTION_LOW) / 512
    rug_counts = (1, 2, 4, 16, 64)
    rug_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rug_counts)))
    density_max = float(np.max(weights / bin_width))
    main_figure, main_axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
    main_axes[0].stairs(
        weights / bin_width,
        np.linspace(ACTION_LOW, ACTION_HIGH, 513),
        color="black",
        linewidth=2.0,
    )
    for row, (num_policy_samples, color) in enumerate(zip(rug_counts, rug_colors)):
        policy_samples = np.asarray(
            results["exact"]["512"][str(num_policy_samples)]["policy_samples"]
        )
        level = -0.07 * density_max * (row + 1)
        main_axes[0].scatter(
            policy_samples,
            np.full_like(policy_samples, level),
            marker="|",
            s=90,
            linewidths=1.5,
            color=color,
            label=f"N={num_policy_samples}",
            clip_on=False,
        )
    main_axes[0].set_ylim(-0.42 * density_max, 1.08 * density_max)
    main_axes[0].set(
        title="Equal-mass sample refinement (M=512)",
        xlabel="Action",
        ylabel="Soft-Q density",
    )
    main_axes[0].legend(frameon=False, fontsize=8, ncol=2)

    for num_proposals in PROPOSAL_COUNTS:
        exact_errors = [
            results["exact"][str(num_proposals)][str(n)]["w2"]
            for n in POLICY_SAMPLE_COUNTS
        ]
        main_axes[1].plot(
            POLICY_SAMPLE_COUNTS,
            exact_errors,
            marker="o",
            color=colors[num_proposals],
            label=f"M={num_proposals}",
        )
    main_axes[1].set_xscale("log", base=2)
    main_axes[1].set_yscale("log")
    main_axes[1].set(
        title="Monotone Wasserstein refinement",
        xlabel="Policy samples N",
        ylabel=r"$W_2(\pi^*_{N,M},\rho_M^Q)$",
        xticks=POLICY_SAMPLE_COUNTS,
    )
    main_axes[1].set_xticklabels(POLICY_SAMPLE_COUNTS)
    main_axes[1].grid(alpha=0.25)
    main_axes[1].legend(frameon=False)
    main_figure.tight_layout()
    main_figure_path = OUTPUT_DIR / "exact_sample_refinement_main.png"
    main_figure.savefig(main_figure_path, dpi=220, bbox_inches="tight")
    plt.close(main_figure)
    return main_figure_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Recompute only the exact finite-sample projection figure.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "setup": {
            "action_bounds": [ACTION_LOW, ACTION_HIGH],
            "temperature": TEMPERATURE,
            "proposal_counts_M": list(PROPOSAL_COUNTS),
            "policy_sample_counts_N": list(POLICY_SAMPLE_COUNTS),
            "seeds": list(SEEDS),
            "train_steps": TRAIN_STEPS,
            "actor_width": ACTOR_WIDTH,
            "actor_hidden_layers": 2,
            "actor_activation": "gelu",
            "latent": "standard_normal_quantile_anchors",
        },
        "exact": {},
        "neural": {},
        "integer_refinement_violations": {},
        "discretization_w2_to_continuous_reference": {},
    }

    reference_actions, reference_weights = make_target(REFERENCE_PROPOSALS)

    for num_proposals in PROPOSAL_COUNTS:
        proposals, target_weights = make_target(num_proposals)
        results["discretization_w2_to_continuous_reference"][str(num_proposals)] = (
            weighted_wasserstein_2(
                proposals,
                target_weights,
                reference_actions,
                reference_weights,
            )
        )
        exact_for_m = {}
        neural_for_m = {}
        exact_errors = {}
        for num_policy_samples in POLICY_SAMPLE_COUNTS:
            target_samples, exact_w2 = optimal_equal_mass_policy_samples(
                proposals,
                target_weights,
                num_policy_samples,
            )
            exact_for_m[str(num_policy_samples)] = {
                "w2": exact_w2,
                "policy_samples": target_samples.tolist(),
            }
            exact_errors[num_policy_samples] = exact_w2
            if not args.exact_only:
                neural_for_m[str(num_policy_samples)] = [
                    run_neural_realization(
                        seed,
                        target_samples,
                        proposals,
                        target_weights,
                    )
                    for seed in SEEDS
                ]
            print(
                f"finished M={num_proposals} N={num_policy_samples} "
                f"exact_W2={exact_w2:.8f}"
            )
        results["exact"][str(num_proposals)] = exact_for_m
        results["neural"][str(num_proposals)] = neural_for_m
        results["integer_refinement_violations"][str(num_proposals)] = (
            verify_integer_refinement(exact_errors)
        )

    if args.exact_only:
        figure_path = None
        main_figure_path = plot_exact_main(results)
    else:
        figure_path, main_figure_path = plot_results(results)
        results["figure"] = str(figure_path)
    results["main_figure"] = str(main_figure_path)
    results_filename = "exact_results.json" if args.exact_only else "results.json"
    results_path = OUTPUT_DIR / results_filename
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({
        "figure": str(figure_path),
        "main_figure": str(main_figure_path),
        "results": str(results_path),
        "integer_refinement_violations": results[
            "integer_refinement_violations"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
