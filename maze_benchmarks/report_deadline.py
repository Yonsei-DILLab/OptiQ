"""Direct-policy and mu-only goal/trajectory comparison with reuse provenance."""
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .visualize_4way import render as render_four
from .report_pointmaze import render_curves, render_medium_hard, render_optiq_modes, render_optiq_overview, render_trajectories
from .evaluation_metrics import removal_from_raw
from .plot_style import TRAJECTORY_ALPHA, TRAJECTORY_COLOR, TRAJECTORY_LINEWIDTH, LEARNING_LINEWIDTH


def record_reporting_corrections(root, manifest):
    corrections = []
    for name in sorted(manifest["runs"]):
        if not name.startswith("pm_"): continue
        folder = root / "runs" / name / "evaluations"
        for path in sorted(folder.glob("*_summary.json")):
            record = json.loads(path.read_text())
            mode = "mu_only" if record["method"] == "optiq" else "policy"
            corrected = removal_from_raw(folder, record, mode)
            original = record[mode]["sr5_removal"]
            corrections.append(dict(run=name, step=record["step"],
                                    mode=mode, original=original, corrected=corrected))
    source = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=Path(__file__).resolve().parents[1], text=True).strip()
    (root / "reporting-corrections.json").write_text(json.dumps(dict(
        reporting_source=source,
        reason="At exactly half the goals reached, all may still be removed; frozen scorer overstated SR5.",
        training_and_raw_files_changed=False, rows=corrections), indent=2) + "\n")


def render_way_metrics(root, manifest, figures, mode):
    colors = {1: "#1f77b4", 3: "#ff7f0e", 5: "#2ca02c", 10: "#9467bd"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    counts_fig, counts_axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)
    for column, (task, count) in enumerate((("4way", 4), ("8way", 8), ("16way", 16))):
        proportions = np.full((4, count + 1), np.nan)
        labels = []
        for row, temperature in enumerate((1, 3, 5, 10)):
            name = f"{task}-optiq-t{temperature}-s0"
            item = manifest["runs"].get(name)
            labels.append(f"T={temperature}" + (" · historical UTD1" if item and item["reused"] else ""))
            if item is None: continue
            summary = item["record"][mode]
            proportions[row] = np.array(summary["goals"] + [summary["failure"]]) / summary["episodes"]
            records = [json.loads(p.read_text()) for p in
                       sorted((root / "runs" / name / "evaluations").glob("*_summary.json"))]
            records = [r for r in records if mode in r]
            steps = [r["step"] / 1000 for r in records]
            for metric, values in enumerate((
                    [r[mode]["success"] for r in records],
                    [sum(v > 0 for v in r[mode]["goals"]) for r in records])):
                axes[metric, column].plot(steps, values, label=labels[-1], color=colors[temperature],
                                          linestyle="--" if item["reused"] else "-", marker="o", ms=4,
                                          linewidth=LEARNING_LINEWIDTH)
        axes[0, column].set(title=task, ylim=(-.02, 1.02), ylabel="Success rate")
        axes[1, column].set(ylim=(-.2, count + .2), ylabel="Goals reached", xlabel="Environment transitions (k)")
        for ax in axes[:, column]:
            ax.set_xlim(0, 1001); ax.grid(alpha=.2)
            handles, legend = ax.get_legend_handles_labels()
            if handles: ax.legend(handles, legend, fontsize=7, loc="best")
        ax = counts_axes[column]
        cmap = plt.colormaps["Blues"].copy(); cmap.set_bad("#eeeeee")
        heat = ax.imshow(proportions, vmin=0., vmax=1., cmap=cmap, aspect="auto")
        ax.set(title=f"{task} · final goal proportions", yticks=range(4), yticklabels=labels,
               xticks=range(count + 1), xticklabels=[str(i) for i in range(count)] + ["Fail"])
        for row in range(4):
            for goal in range(count + 1):
                value = proportions[row, goal]
                if np.isfinite(value):
                    ax.text(goal, row, f"{100*value:.1f}", ha="center", va="center", fontsize=8,
                            color="white" if value > .55 else "black")
    label = "direct policy including conditional sigma" if mode == "policy" else "random-z mu-only"
    fig.suptitle(f"OptiQ · {label} · seed0\nHistorical T1 uses a different update budget; only archived complete runs shown")
    fig.savefig(figures / f"way_learning_curves_{mode}.png", dpi=160)
    plt.close(fig)
    counts_fig.colorbar(heat, ax=list(counts_axes), label="Fraction of all evaluation episodes", shrink=.7)
    counts_fig.suptitle(f"OptiQ ~1M · {label} · seed0 · values in percent\nGrey rows pending; failures included; goal IDs match the saved environment")
    counts_fig.savefig(figures / f"way_goal_proportions_{mode}.png", dpi=160)
    plt.close(counts_fig)


def render(root: Path):
    manifest = json.loads((root / "archive-manifest.json").read_text())
    record_reporting_corrections(root, manifest)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    supplement = figures / "supplementary_sigma"
    supplement.mkdir(exist_ok=True)
    for mode in ("mu_only", "policy"):
        mode_figures = figures if mode == "mu_only" else supplement
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
                    for xy in trajectories[:100]:
                        ax.plot(xy[:, 0], xy[:, 1], color=TRAJECTORY_COLOR,
                                alpha=TRAJECTORY_ALPHA, lw=TRAJECTORY_LINEWIDTH,
                                solid_capstyle="round")
                ax.scatter(goals[:, 0], goals[:, 1], c="#2caf37", marker="*", s=70, zorder=3)
                ax.scatter([0], [0], c="red", s=20, zorder=4)
                suffix = " · reused UTD1" if item["reused"] else ""
                ax.set_title(f"{task} T={temp}{suffix}\n{summary['success']:.1%} success · {summary['reachable_goals']}/{len(goals)} goals")
        label = "sampled policy including conditional sigma" if mode == "policy" else "random-z mu-only, no conditional sigma"
        fig.suptitle(f"OptiQ ~1M transitions, seed0 · {label}\nFirst100 trajectories; metrics use every evaluation episode. Reused T1 runs have different update budgets.")
        fig.savefig(mode_figures / f"way_trajectories_{mode}.png", dpi=150)
        plt.close(fig)
        render_way_metrics(root, manifest, mode_figures, mode)
    entries = []
    for temp in (1, 3, 5, 10):
        name = f"4way-optiq-t{temp}-s0"
        if name in manifest["runs"]:
            item = manifest["runs"][name]; e = root / "runs" / name / "evaluations"
            entries.append((f"OptiQ T={temp} · ~1M", e / f"{item['steps']:09d}_policy.npz", e / f"{item['steps']:09d}_probe.npz"))
    if entries: render_four(entries, supplement / "4way_policy_and_learned_q.png")
    render_curves(root, figures / "pointmaze_learning_curves.png")
    render_medium_hard(root, figures / "pointmaze_medium_hard.png")
    render_optiq_overview(root, figures / "pointmaze_optiq_overview.png")
    for maze in ("simple", "medium", "hard"):
        render_trajectories(root, maze, figures / f"pointmaze_{maze}.png")
        render_optiq_modes(root, maze, supplement / f"pointmaze_{maze}_optiq_policy_vs_mu.png")
    lines = ["# 1M experiment results", "", f"Verified complete: {manifest['complete']}/33.", "",
             "All runs use seed0. Success is not mode coverage. Way T1 reused8/16 results use UTD1; new results use UTD0.0625 and cannot isolate temperature effects.", "",
             "Removal SR5 curves are recomputed from raw five-episode groups to correct a frozen scorer boundary error. See reporting-corrections.json; training, success/goal counts, trajectories and raw archives are unchanged.", "",
             "OptiQ primary evaluation: fresh random-z mu-only, without conditional sigma. Baselines: native samples. Historical sigma-included figures are supplementary; old Q probes are not mu-only. Missing obstacle-mu results are omitted.", "",
             "|Run|Success|Goals visited|Goal counts|", "|---|---:|---:|---|"]
    for name, result in sorted(manifest["runs"].items()):
        mode = "mu_only" if "optiq" in name else "policy"
        r = result["record"][mode]
        lines.append(f"|{name}{' (reused)' if result['reused'] else ''}|{r['success']:.1%}|{r['reachable_goals']}/{len(r['goals'])}|{r['goals']}|")
    (root / "REPORT.md").write_text("\n".join(lines)+"\n")
