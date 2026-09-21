"""Summarize bounded Humanoid screens without changing their logs or processes."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v2_improvement"


def read(directory):
    with np.load(next(directory.glob("eval/*/evaluations.npz"))) as data:
        steps = data["timesteps"].copy()
        means = data["results"].mean(axis=1)
    cfg = json.loads((directory / "config.json").read_text())
    return steps, means, cfg


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    reference = next((ROOT / "outputs/v2_screen").glob("*optiq-reference*"))
    rt, rv, _ = read(reference)
    rows = []
    for index, folder in enumerate(("v2_screen", "v2_conditional_screen")):
        for directory in sorted((ROOT / "outputs" / folder).glob("humanoid-*")):
            t, v, cfg = read(directory)
            actor = cfg["alg"]["actor"]
            if "reference" in directory.name:
                label, color, style = "OptiQ reference, seed 0", "#303030", "-"
            elif folder == "v2_screen":
                nll = "otnll" in directory.name
                label = f"{'Full OT NLL' if nll else 'Argmax MSE'}, T={actor['temperature']}"
                color = "#176ab0" if nll and actor["temperature"] == .1 else "#d17c15" if nll else "#b53040"
                style = "-"
            else:
                label = f"Conditional, T={actor['temperature']}, seed {cfg['seed']}"
                color = "#176ab0" if actor["temperature"] == .1 else "#d17c15"
                style = "-" if cfg["seed"] == 0 else "--"
            axes[index].plot(t / 1000, v, label=label, color=color, linestyle=style, linewidth=1.6)
            rows.append({"run": directory.name, "seed": cfg["seed"],
                "steps": int(t[-1]), "final_eval_return": float(v[-1]),
                "last_three_eval_mean": float(v[-3:].mean()),
                "completed": (directory / "completed.json").exists()})
        if index:
            axes[index].plot(rt / 1000, rv, label="OptiQ reference, seed 0", color="#303030", linewidth=2)
        axes[index].set_title(("Projection and temperature", "Learned conditional proposal")[index])
        axes[index].set_xlabel("Environment steps (thousands)")
        axes[index].set_xlim(0, 100)
        axes[index].grid(alpha=.2)
        axes[index].legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Mean evaluation return")
    fig.suptitle("Humanoid-v4: bounded screening, no additional uniform exploration")
    fig.text(.5, .01, "3 stochastic episodes per evaluation; curves are not seed confidence intervals. "
             "The OptiQ reference is seed 0 only.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / "v2_screen_comparison.png", dpi=180)
    fig.savefig(OUT / "v2_screen_comparison.pdf")
    (OUT / "v2_screen_comparison.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
