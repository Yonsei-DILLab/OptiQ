"""Aggregate completed paired HalfCheetah learning curves and final returns."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("direct", "smem_tr")
LABELS = {"direct": "Direct GMM/TRG", "smem_tr": "SMEM+TR"}
COLORS = {"direct": "#2878a5", "smem_tr": "#c84b31"}


def load_run(campaign, method, seed):
    candidates = list((campaign/"outputs").glob(
        f"halfcheetah-{method}-seed{seed}_*"
    ))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one output for {method} seed {seed}: {candidates}")
    root = candidates[0]
    result = {"method": method, "seed": seed, "root": str(root)}
    completed = json.loads((root/"completed.json").read_text())
    result["wandb_url"] = completed["wandb_url"]
    for mode in ("stochastic_z", "zero_z"):
        files = list(root.rglob(f"evaluations_{mode}.npz"))
        if len(files) != 1:
            raise RuntimeError(f"missing {mode} evaluation in {root}")
        with np.load(files[0]) as data:
            steps = data["timesteps"].astype(float)
            episode_returns = data["results"].astype(float)
        if int(steps[-1]) != 1_000_000 or episode_returns.shape[1] != 10:
            raise RuntimeError(f"incomplete evaluation {files[0]}")
        means = episode_returns.mean(axis=1)
        result[mode] = {
            "steps": steps.tolist(),
            "mean_returns": means.tolist(),
            "final_return": float(means[-1]),
            "auc": float(np.trapz(means, steps)/(steps[-1]-steps[0])),
        }
    state = json.loads((campaign/"state"/f"halfcheetah-{method}-seed{seed}.json").read_text())
    result["slurm_job_id"] = state["slurm_job_id"]
    result["started_utc"] = state["started_utc"]
    result["finished_utc"] = state["finished_utc"]
    return result


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    output = args.campaign/"analysis"
    output.mkdir(exist_ok=True)
    runs = [load_run(args.campaign, method, seed) for seed in range(3) for method in METHODS]
    summary = {}
    for method in METHODS:
        selected = [run for run in runs if run["method"] == method]
        summary[method] = {}
        for mode in ("stochastic_z", "zero_z"):
            summary[method][mode] = {
                metric: dict(zip(("mean", "sample_std"), mean_std([
                    run[mode][metric] for run in selected
                ])))
                for metric in ("final_return", "auc")
            }
    paired = {}
    for mode in ("stochastic_z", "zero_z"):
        for metric in ("final_return", "auc"):
            differences = [
                next(r for r in runs if r["method"] == "smem_tr" and r["seed"] == seed)[mode][metric]
                - next(r for r in runs if r["method"] == "direct" and r["seed"] == seed)[mode][metric]
                for seed in range(3)
            ]
            paired[f"{mode}_{metric}_smem_minus_direct"] = {
                "values": differences,
                "mean": float(np.mean(differences)),
                "sample_std": float(np.std(differences, ddof=1)),
            }
    payload = {"runs": runs, "summary": summary, "paired": paired}
    (output/"results.json").write_text(json.dumps(payload, indent=2)+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for axis, mode in zip(axes, ("stochastic_z", "zero_z")):
        for method in METHODS:
            selected = [run for run in runs if run["method"] == method]
            steps = np.asarray(selected[0][mode]["steps"])
            curves = np.asarray([run[mode]["mean_returns"] for run in selected])
            mean, std = curves.mean(0), curves.std(0, ddof=1)
            axis.plot(steps, mean, label=LABELS[method], color=COLORS[method])
            axis.fill_between(steps, mean-std, mean+std, color=COLORS[method], alpha=.18)
        axis.set_title(mode.replace("_", " "))
        axis.set_xlabel("environment steps")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("evaluation return (mean ± seed SD)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output/"learning_curves.png", dpi=180)
    plt.close(fig)

    lines = [
        "# HalfCheetah Direct GMM/TRG 대 SMEM+TR", "",
        "모든 수치는 동일한 `HalfCheetah-v4` 1M-step protocol의 시드 0/1/2 실측값이다.", "",
        "| 방법 | stochastic final | stochastic AUC | zero-z final | zero-z AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = []
        for mode, metric in (("stochastic_z", "final_return"), ("stochastic_z", "auc"),
                             ("zero_z", "final_return"), ("zero_z", "auc")):
            item = summary[method][mode][metric]
            values.append(f"{item['mean']:.1f} ± {item['sample_std']:.1f}")
        lines.append(f"| {LABELS[method]} | " + " | ".join(values) + " |")
    lines += ["", "## Paired SMEM−Direct differences", ""]
    for name, item in paired.items():
        lines.append(f"- `{name}`: {item['mean']:.2f} ± {item['sample_std']:.2f}; seeds "
                     + ", ".join(f"{value:.2f}" for value in item["values"]))
    lines += ["", "## Runs", "", "| method | seed | SLURM | W&B |", "|---|---:|---:|---|"]
    for run in runs:
        lines.append(f"| {LABELS[run['method']]} | {run['seed']} | {run['slurm_job_id']} | {run['wandb_url']} |")
    (output/"REPORT_KO.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"summary": summary, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
