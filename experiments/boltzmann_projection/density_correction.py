"""Consistency of proposal-density-corrected Boltzmann weighting."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm


OUTPUT_DIR = Path("outputs/boltzmann_projection/density_correction")
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
TEMPERATURE = 1.0
SAMPLE_COUNTS = (64, 128, 256, 512, 1024, 2048)
SEEDS = tuple(range(64))
REFERENCE_SIZE = 8_192
HISTOGRAM_BINS = 120
PLOT_SAMPLE_COUNT = 512


def oracle_q(actions):
    actions = np.asarray(actions, dtype=np.float64)
    modes = np.stack(
        [
            np.log(0.58) - 0.5 * ((actions + 0.85) / 0.22) ** 2,
            np.log(0.42) - 0.5 * ((actions - 0.75) / 0.34) ** 2,
        ],
        axis=0,
    )
    return logsumexp(modes, axis=0)


def normalized_weights(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()


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
    while i < len(samples_a) and j < len(samples_b):
        transported = min(remaining_a, remaining_b)
        squared_cost += transported * (samples_a[i] - samples_b[j]) ** 2
        remaining_a -= transported
        remaining_b -= transported
        if remaining_a <= 1.0e-15:
            i += 1
            if i < len(samples_a):
                remaining_a = weights_a[i]
        if remaining_b <= 1.0e-15:
            j += 1
            if j < len(samples_b):
                remaining_b = weights_b[j]
    return float(np.sqrt(max(squared_cost, 0.0)))


def truncated_normal_density(actions, mean, std):
    lower = (ACTION_LOW - mean) / std
    upper = (ACTION_HIGH - mean) / std
    return truncnorm.pdf(actions, lower, upper, loc=mean, scale=std)


def sample_truncated_normal(rng, count, mean, std):
    lower = (ACTION_LOW - mean) / std
    upper = (ACTION_HIGH - mean) / std
    return truncnorm.rvs(
        lower,
        upper,
        loc=mean,
        scale=std,
        size=count,
        random_state=rng,
    )


@dataclass(frozen=True)
class Proposal:
    name: str
    title: str
    sampler: object
    density: object


def make_proposals():
    uniform_density = 1.0 / (ACTION_HIGH - ACTION_LOW)

    def sample_uniform(rng, count):
        return rng.uniform(ACTION_LOW, ACTION_HIGH, size=count)

    def density_uniform(actions):
        return np.full_like(np.asarray(actions), uniform_density, dtype=np.float64)

    def sample_shifted(rng, count):
        return sample_truncated_normal(rng, count, mean=-0.65, std=0.72)

    def density_shifted(actions):
        return truncated_normal_density(actions, mean=-0.65, std=0.72)

    mixture_weights = np.asarray([0.68, 0.32])
    mixture_means = np.asarray([-0.95, 0.45])
    mixture_stds = np.asarray([0.38, 0.52])

    def sample_mixture(rng, count):
        components = rng.choice(2, size=count, p=mixture_weights)
        samples = np.empty(count, dtype=np.float64)
        for component in range(2):
            mask = components == component
            samples[mask] = sample_truncated_normal(
                rng,
                int(mask.sum()),
                mixture_means[component],
                mixture_stds[component],
            )
        return samples

    def density_mixture(actions):
        actions = np.asarray(actions)
        return sum(
            weight * truncated_normal_density(actions, mean, std)
            for weight, mean, std in zip(
                mixture_weights,
                mixture_means,
                mixture_stds,
            )
        )

    return (
        Proposal("uniform", "Uniform proposal", sample_uniform, density_uniform),
        Proposal(
            "shifted_truncnorm",
            "Shifted truncated Gaussian",
            sample_shifted,
            density_shifted,
        ),
        Proposal(
            "policy_mixture",
            "Policy-like Gaussian mixture",
            sample_mixture,
            density_mixture,
        ),
    )


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def run_experiment():
    width = (ACTION_HIGH - ACTION_LOW) / REFERENCE_SIZE
    target_actions = ACTION_LOW + (np.arange(REFERENCE_SIZE) + 0.5) * width
    target_weights = normalized_weights(oracle_q(target_actions) / TEMPERATURE)
    proposal_grid = np.linspace(ACTION_LOW, ACTION_HIGH, 2_001)

    results = {
        "configuration": {
            "action_bounds": [ACTION_LOW, ACTION_HIGH],
            "temperature": TEMPERATURE,
            "sample_counts": list(SAMPLE_COUNTS),
            "seeds": list(SEEDS),
            "reference_size": REFERENCE_SIZE,
        },
        "proposals": {},
    }
    histogram_edges = np.linspace(
        ACTION_LOW,
        ACTION_HIGH,
        HISTOGRAM_BINS + 1,
    )
    histogram_centers = 0.5 * (histogram_edges[:-1] + histogram_edges[1:])
    histogram_width = histogram_edges[1] - histogram_edges[0]

    for proposal in make_proposals():
        proposal_results = {
            "title": proposal.title,
            "sample_counts": {},
            "proposal_grid_density": proposal.density(proposal_grid).tolist(),
        }
        density_histograms = {"naive": [], "corrected": []}
        for count in SAMPLE_COUNTS:
            run_metrics = {"naive": [], "corrected": []}
            for seed in SEEDS:
                rng = np.random.default_rng(1_000_000 * count + seed)
                samples = proposal.sampler(rng, count)
                log_density = np.log(np.maximum(proposal.density(samples), 1.0e-300))
                logits = oracle_q(samples) / TEMPERATURE
                method_weights = {
                    "naive": normalized_weights(logits),
                    "corrected": normalized_weights(logits - log_density),
                }
                if proposal.name == "uniform":
                    np.testing.assert_allclose(
                        method_weights["naive"],
                        method_weights["corrected"],
                        atol=1.0e-12,
                        rtol=1.0e-12,
                    )
                for method, weights in method_weights.items():
                    w2 = weighted_wasserstein_2(
                        samples,
                        weights,
                        target_actions,
                        target_weights,
                    )
                    ess = 1.0 / np.sum(np.square(weights))
                    run_metrics[method].append(
                        {
                            "w2": w2,
                            "ess": float(ess),
                            "ess_fraction": float(ess / count),
                            "top1": float(weights.max()),
                        }
                    )
                    if count == PLOT_SAMPLE_COUNT:
                        mass, _ = np.histogram(
                            samples,
                            bins=histogram_edges,
                            weights=weights,
                        )
                        density_histograms[method].append(mass / histogram_width)

            proposal_results["sample_counts"][str(count)] = {
                method: {
                    metric: summarize([entry[metric] for entry in entries])
                    for metric in ("w2", "ess", "ess_fraction", "top1")
                }
                for method, entries in run_metrics.items()
            }

        proposal_results["histogram_centers"] = histogram_centers.tolist()
        proposal_results["recovered_density"] = {
            method: {
                "mean": np.mean(histograms, axis=0).tolist(),
                "std": np.std(histograms, axis=0).tolist(),
            }
            for method, histograms in density_histograms.items()
        }
        results["proposals"][proposal.name] = proposal_results

    results["reference"] = {
        "actions": target_actions.tolist(),
        "weights": target_weights.tolist(),
        "proposal_grid": proposal_grid.tolist(),
    }
    return results


def plot_results(
    results,
    proposal_names=("uniform", "shifted_truncnorm", "policy_mixture"),
    output_name="density_correction_consistency.png",
):
    proposal_grid = np.asarray(results["reference"]["proposal_grid"])
    target_actions = np.asarray(results["reference"]["actions"])
    target_weights = np.asarray(results["reference"]["weights"])
    target_density = target_weights / (
        target_actions[1] - target_actions[0]
    )
    colors = {"naive": "#2f6fb0", "corrected": "#d1495b"}

    figure_width = 14.2 if len(proposal_names) == 3 else 10.2
    figure, axes = plt.subplots(
        3,
        len(proposal_names),
        figsize=(figure_width, 9.2),
        squeeze=False,
    )
    for column, proposal_name in enumerate(proposal_names):
        entry = results["proposals"][proposal_name]
        axes[0, column].plot(
            proposal_grid,
            entry["proposal_grid_density"],
            color="#477a5b",
            linewidth=2.2,
            label=r"Proposal $q_F$",
        )
        axes[0, column].plot(
            target_actions,
            target_density,
            color="black",
            linewidth=2.0,
            label=r"Target $\rho_Q$",
        )
        axes[0, column].set_title(entry["title"], fontsize=12)
        axes[0, column].set_ylabel("Density")

        centers = np.asarray(entry["histogram_centers"])
        for method in ("naive", "corrected"):
            density = np.asarray(entry["recovered_density"][method]["mean"])
            std = np.asarray(entry["recovered_density"][method]["std"])
            axes[1, column].plot(
                centers,
                density,
                color=colors[method],
                linewidth=2.0,
                label=method.capitalize(),
            )
            axes[1, column].fill_between(
                centers,
                np.maximum(density - std, 0.0),
                density + std,
                color=colors[method],
                alpha=0.12,
                linewidth=0,
            )
        axes[1, column].plot(
            target_actions,
            target_density,
            color="black",
            linewidth=1.8,
            linestyle="--",
            label=r"Target $\rho_Q$",
        )
        axes[1, column].set_ylabel(f"Recovered density\n($M={PLOT_SAMPLE_COUNT}$)")

        for method in ("naive", "corrected"):
            means = []
            stds = []
            for count in SAMPLE_COUNTS:
                summary = entry["sample_counts"][str(count)][method]["w2"]
                means.append(summary["mean"])
                stds.append(summary["std"])
            means = np.asarray(means)
            stds = np.asarray(stds)
            axes[2, column].plot(
                SAMPLE_COUNTS,
                means,
                marker="o",
                markersize=4.5,
                color=colors[method],
                linewidth=2.0,
                label=method.capitalize(),
            )
            axes[2, column].fill_between(
                SAMPLE_COUNTS,
                np.maximum(means - stds, 0.0),
                means + stds,
                color=colors[method],
                alpha=0.13,
                linewidth=0,
            )
        axes[2, column].set_xscale("log", base=2)
        axes[2, column].set_xlabel("Number of proposal samples $M$")
        axes[2, column].set_ylabel(r"$W_2$ to $\rho_Q$")
        axes[2, column].grid(alpha=0.25)

    for row in range(3):
        axes[row, 0].legend(frameon=False, fontsize=9)
    for axis in axes[:2].flat:
        axis.set_xlim(ACTION_LOW, ACTION_HIGH)
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Proposal-density correction recovers the same Boltzmann target",
        fontsize=15,
        y=0.995,
    )
    figure.tight_layout()
    output = OUTPUT_DIR / output_name
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_experiment()
    result_path = OUTPUT_DIR / "density_correction_consistency.json"
    result_path.write_text(json.dumps(results, indent=2))
    figure_path = plot_results(results)
    main_figure_path = plot_results(
        results,
        proposal_names=("uniform", "shifted_truncnorm"),
        output_name="density_correction_uniform_vs_shifted.png",
    )

    for proposal_name, entry in results["proposals"].items():
        largest = entry["sample_counts"][str(SAMPLE_COUNTS[-1])]
        print(
            proposal_name,
            "naive_w2=",
            f"{largest['naive']['w2']['mean']:.4f}",
            "corrected_w2=",
            f"{largest['corrected']['w2']['mean']:.4f}",
            "corrected_ess=",
            f"{largest['corrected']['ess']['mean']:.1f}",
        )
    print(figure_path)
    print(main_figure_path)
    print(result_path)


if __name__ == "__main__":
    main()
