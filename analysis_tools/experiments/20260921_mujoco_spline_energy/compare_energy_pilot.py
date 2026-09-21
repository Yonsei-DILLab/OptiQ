"""Local paired-prefix report; does not submit jobs or sync W&B."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue  # Allows a live SLURM log / final partially written row.
        if isinstance(value, dict) and "env_steps" in value:
            result.append(value)
    return result


def summarize(values):
    if not values:
        return None
    x = np.array([v["env_steps"] for v in values])
    y = np.array([v["eval/mean_reward"] for v in values])
    return {
        "steps": int(x[-1]),
        "last_mean_reward": float(y[-1]),
        "last_episode_std": values[-1]["eval/std_reward"],
        "last_mean_length": values[-1]["eval/mean_length"],
        "reward_auc_per_step": float(np.trapz(y, x) / x[-1]) if x[-1] else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=Path(
        "/scratch2/gsmin2024/research/optiq_spline_energy_bellman_20260922"))
    args = parser.parse_args()
    campaign = args.campaign
    manifest = json.loads((campaign / "manifest.json").read_text())
    baseline = Path(manifest["baseline"]["output"])
    pilot = Path(manifest["output"])
    limit = manifest["total_steps"]
    ref = [r for r in rows(baseline / "evaluations.jsonl") if r["env_steps"] <= limit]
    new = [r for r in rows(pilot / "evaluations.jsonl") if r["env_steps"] <= limit]
    protocol_match = None
    if (pilot / "config.json").exists():
        bc = json.loads((baseline / "config.json").read_text())
        pc = json.loads((pilot / "config.json").read_text())
        keys = list(manifest["hyperparameters"]) + [
            "env", "seed", "packages", "behavior_uniform_probability",
            "time_limit_bootstrap", "trainable_parameters"]
        mismatch = {k: [bc[k], pc[k]] for k in keys if bc[k] != pc[k]}
        if mismatch:
            raise ValueError(f"Comparison protocol differs: {mismatch}")
        protocol_match = True
    common_steps = sorted({r["env_steps"] for r in ref} & {r["env_steps"] for r in new})
    paired_ref = [r for r in ref if r["env_steps"] in common_steps]
    paired_new = [r for r in new if r["env_steps"] in common_steps]
    status = "completed" if (pilot / "COMPLETE.json").exists() else (
        "running_or_partial" if new else "awaiting_evaluation")
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "protocol_match": protocol_match,
        "baseline_100k_prefix": summarize(ref),
        "baseline_at_common_steps": summarize(paired_ref),
        "relative_energy_at_common_steps": summarize(paired_new),
        "paired_evaluations": [{
            "env_steps": b["env_steps"], "huber_reward": b["eval/mean_reward"],
            "relative_energy_reward": n["eval/mean_reward"],
            "difference": n["eval/mean_reward"] - b["eval/mean_reward"],
        } for b, n in zip(paired_ref, paired_new)],
        "limitations": "One training seed. Episode variability is not across-seed uncertainty. "
                        "100k steps is an early diagnostic, not a 1M result. "
                        "Empirical TD error is not a uniform Bellman certificate.",
        "training_commit": manifest["git_commit"],
    }
    launch = json.loads((campaign / "launch.json").read_text())
    baseline_log = Path(manifest["baseline"]["training_log"])
    pilot_log = campaign / f"logs/pilot-{launch['job_id']}.out"
    ref_td = [r for r in rows(baseline_log) if 5000 < r["env_steps"] <= limit]
    new_td = [r for r in rows(pilot_log) if 5000 < r["env_steps"] <= limit]
    # These are minibatch diagnostics, not expected or uniform Bellman errors.
    common_end = min([r[-1]["env_steps"] for r in [ref_td, new_td] if r], default=0)
    for name, values in [("huber", ref_td), ("relative_energy", new_td)]:
        tail = [r for r in values if common_end - 10000 < r["env_steps"] <= common_end]
        if tail:
            result[f"{name}_last_10k_sample_diagnostics"] = {
                key: float(np.mean([r[key] for r in tail]))
                for key in ["train/abs_td", "train/q_mean", "train/v_mean",
                            "train/target_mean", "train/grad_norm_preclip"]}
    out = campaign / "reports"
    out.mkdir(exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for values, name, color in [(ref, "Spline + Huber", "#596579"),
                                 (new, "Spline + relative energy", "#007E87")]:
        if values:
            x = [r["env_steps"] for r in values]
            axes[0].plot(x, [r["eval/mean_reward"] for r in values], label=name, color=color)
            axes[1].plot(x, [r["eval/mean_length"] for r in values], label=name, color=color)
    for values, name, color in [(ref_td, "Spline + Huber", "#596579"),
                                 (new_td, "Spline + relative energy", "#007E87")]:
        if values:
            axes[2].plot([r["env_steps"] for r in values],
                         [r["train/abs_td"] for r in values], label=name, color=color, alpha=.8)
    for ax, label in zip(axes, ["Evaluation mean return", "Evaluation mean episode length",
                              "Minibatch mean absolute TD error"]):
        ax.set(xlabel="Environment steps", ylabel=label, xlim=(0, limit))
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].legend()
    fig.suptitle(f"Ant-v4, seed 0, 10 evaluation episodes | {status}")
    fig.text(.5, .01, "Matched step budgets; one training seed; no across-seed confidence interval.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out / "ant_100k_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != "paired_evaluations"}, indent=2))


if __name__ == "__main__":
    main()
