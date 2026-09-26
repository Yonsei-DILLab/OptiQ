"""Direct-policy and mu-only goal/trajectory comparison with reuse provenance."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .visualize_4way import render as render_four
from .report_pointmaze import render_curves, render_medium_hard, render_optiq_modes, render_trajectories


def render(root: Path):
    manifest = json.loads((root / "archive-manifest.json").read_text())
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for mode in ("policy", "mu_only"):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
        for i, task in enumerate(("4way", "8way", "16way")):
            for j, temp in enumerate((1, 3, 5, 10)):
                ax = axes[i, j]
                name = f"{task}-optiq-t{temp}-s0"
                item = manifest["runs"].get(name)
                ax.set(xlim=(-7.5, 7.5), ylim=(-7.5, 7.5), aspect="equal")
                if item is None:
                    ax.set_title(f"{task} T={temp} · pending"); continue
                folder = root / "runs" / name
                config = json.loads((folder / "config.json").read_text())
                goals = np.array([[5, 0], [-5, 0], [0, 5], [0, -5]]) if task == "4way" else np.asarray(config["geometry"]["goal_positions"])
                summary = item["record"][mode]
                path = folder / "evaluations" / f"{item['steps']:09d}_{mode}.npz"
                with np.load(path) as raw:
                    trajectories = raw["xy"]
                    for xy in trajectories[:100]: ax.plot(xy[:, 0], xy[:, 1], color="#d329da", alpha=.3, lw=.65)
                ax.scatter(goals[:, 0], goals[:, 1], c="#2caf37", marker="*", s=70, zorder=3)
                ax.scatter([0], [0], c="red", s=20, zorder=4)
                suffix = " · reused UTD1" if item["reused"] else ""
                ax.set_title(f"{task} T={temp}{suffix}\n{summary['success']:.1%} success · {summary['reachable_goals']}/{len(goals)} goals")
        label = "sampled policy including conditional sigma" if mode == "policy" else "random-z mu-only, no conditional sigma"
        fig.suptitle(f"OptiQ ~1M transitions, seed0 · {label}\nFirst100 trajectories; metrics use every evaluation episode. Reused T1 runs have different update budgets.")
        fig.savefig(figures / f"way_trajectories_{mode}.png", dpi=150)
        plt.close(fig)
    entries = []
    for temp in (1, 3, 5, 10):
        name = f"4way-optiq-t{temp}-s0"
        if name in manifest["runs"]:
            item = manifest["runs"][name]; e = root / "runs" / name / "evaluations"
            entries.append((f"OptiQ T={temp} · ~1M", e / f"{item['steps']:09d}_policy.npz", e / f"{item['steps']:09d}_probe.npz"))
    if entries: render_four(entries, figures / "4way_policy_and_learned_q.png")
    render_curves(root, figures / "pointmaze_learning_curves.png")
    render_medium_hard(root, figures / "pointmaze_medium_hard.png")
    for maze in ("simple", "medium", "hard"):
        render_trajectories(root, maze, figures / f"pointmaze_{maze}.png")
        render_optiq_modes(root, maze, figures / f"pointmaze_{maze}_optiq_policy_vs_mu.png")
    lines = ["# 1M experiment results", "", f"Verified complete: {manifest['complete']}/33.", "",
             "All runs use seed0. Success is not mode coverage. Way T1 reused8/16 results use UTD1; new results use UTD0.0625 and cannot isolate temperature effects.", "",
             "|Run|Success|Goals visited|Goal counts|", "|---|---:|---:|---|"]
    for name, result in sorted(manifest["runs"].items()):
        r = result["record"]["policy"]
        lines.append(f"|{name}{' (reused)' if result['reused'] else ''}|{r['success']:.1%}|{r['reachable_goals']}/{len(r['goals'])}|{r['goals']}|")
    (root / "REPORT.md").write_text("\n".join(lines)+"\n")
