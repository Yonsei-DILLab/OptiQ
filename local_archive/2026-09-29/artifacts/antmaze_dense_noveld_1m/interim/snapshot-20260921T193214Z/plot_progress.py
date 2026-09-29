"""Plot saved periodic evaluation summaries without touching live training."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, FuncFormatter

root = Path(__file__).resolve().parent
source = root / "evaluation-history.json"
snapshot = json.loads(source.read_text())
runs = {job: histories for host in snapshot["hosts"].values()
        for job, histories in host["runs"].items()}
colors = {"optiq": "#078679", "sac": "#cc7338", "mfpo": "#4b65b5", "meow": "#995d9c"}
names = {"optiq": "OptiQ", "sac": "SAC", "mfpo": "MFPO", "meow": "MEOW"}
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharey=True)
for row, task in enumerate(("v1", "v2")):
    max_step = max(p["step"] for method in colors
                   for history in runs[f"{task}-{method}-s0"].values() for p in history)
    for col, mode in enumerate(("native", "policy")):
        ax = axes[row, col]
        for method, color in colors.items():
            points = runs[f"{task}-{method}-s0"][f"history-{mode}-natural.json"]
            assert all(p["episodes"] == 10 and p["training_seed"] == 0 for p in points)
            assert all(p["evaluation_mode"] == mode for p in points)
            ax.plot([p["step"] for p in points], [p["success_rate"] for p in points],
                    color=color, marker="o", markersize=4, linewidth=1.8, label=names[method])
        ax.set(title=f"AntMaze {task} · " + ("Native evaluation" if mode == "native" else "Direct stochastic policy"),
               xlabel="Environment interactions", ylabel="Goal success (10 rollouts)",
               ylim=(-.035, 1.035), xlim=(0, max_step * 1.035))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("AntMaze · dense reward + NovelD · interim evaluations", fontsize=16, y=.985)
fig.legend([Line2D([0], [0], color=c, marker="o") for c in colors.values()],
           list(names.values()), loc="upper center", ncol=4, bbox_to_anchor=(.5, .952), frameon=False)
fig.text(.5, .063, "One training seed (0); 10 rollouts per point; no smoothing. Lines end at the latest available evaluation.",
         ha="center", fontsize=9)
fig.text(.5, .040, "Native: OptiQ random-z mu-only / SAC mean / MFPO Q-best-of-10 / MEOW prior center.",
         ha="center", fontsize=9)
fig.text(.5, .017, "Direct policy includes OptiQ conditional sigma. No added DACER noise or NovelD reward during evaluation. v3/v4 queued.",
         ha="center", fontsize=8.5)
fig.tight_layout(rect=(0, .092, 1, .91))
for extension in ("png", "pdf"):
    fig.savefig(root / f"success-progress.{extension}", dpi=180)
plt.close(fig)
proof = dict(collected_at=snapshot["collected_at"], source_commit=snapshot["training_source_commit"],
             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             all_experiments_complete=False, training_seed=0, episodes_per_point=10,
             note="Interim curves; differing training budgets. No evidence of multiple successful routes from a single success.")
(root / "success-progress-provenance.json").write_text(json.dumps(proof, indent=2) + "\n")
