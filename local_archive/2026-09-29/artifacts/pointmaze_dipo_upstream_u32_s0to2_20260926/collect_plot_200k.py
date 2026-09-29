"""Read-only collection of corrected DIPO 200k rollouts; no training changes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

BASE = Path(__file__).resolve().parent / "trajectory_200k"
REPO = Path("/Users/yunheechan/Documents/ChatGPT/OptiQ/tmp/pointmaze-multiseed-worktree")
sys.path.insert(0, str(REPO))
STEP = 200704
CAMPAIGN = "pointmaze-dipo-upstream-u32-s0to2-1m-20260926"
OTHER = "maze-dipo-upstream-u32-s0-1m-20260926"
JOBS = [("vast-heechan-6", OTHER, "simple", 0)] + [
    ("vast-heechan-199" if seed == 2 else "vast-heechan-46", CAMPAIGN, maze, seed)
    for maze in ("medium", "hard") for seed in range(3)
]


def collect(job):
    host, campaign, maze, seed = job
    name = f"pm_{maze}-dipo-upstream-u32-s{seed}"
    remote = f"/home/heechan/optiq-experiments/{campaign}/runs/{name}"
    destination = BASE / name
    files = ["config.json", f"evaluations/{STEP:09d}_summary.json",
             f"evaluations/{STEP:09d}_policy.npz"]
    script = ("import pathlib,json,hashlib\n"
              f"r=pathlib.Path({remote!r})\n"
              f"print(json.dumps({{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in {files!r}}}))\n")
    output = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                             "python3 -"], input=script, text=True, capture_output=True, check=True)
    hashes = json.loads(output.stdout)
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{host}:{remote}/{relative}",
                        str(target)], check=True, capture_output=True)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[relative]
    return dict(host=host, campaign=campaign, maze=maze, seed=seed, name=name,
                remote=remote, sha256=hashes)


BASE.mkdir(parents=True, exist_ok=True)
if "--render-only" in sys.argv:
    records = json.loads((BASE / "figure-manifest.json").read_text())["records"]
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(collect, JOBS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from maze_benchmarks.visualize_pointmaze import plot_map, plot_rollouts

for record in records:
    folder = BASE / record["name"]
    summary = json.loads((folder / f"evaluations/{STEP:09d}_summary.json").read_text())
    with np.load(folder / f"evaluations/{STEP:09d}_policy.npz") as raw:
        goal_ids = raw["goal_ids"]
        count = np.bincount(goal_ids[goal_ids >= 0], minlength=8 if record["maze"] == "hard" else 4)
        assert len(goal_ids) == 200
        assert count.tolist() == summary["policy"]["goals"]
        assert float(np.mean(goal_ids >= 0)) == summary["policy"]["success"]
    assert summary["step"] == STEP and summary["updates"] == 3008
    assert summary["source_commit"] in ("041738202b5e6bcf765a9a807e917afd2778608a",
                                         "5f4b809d0e95899eef2435dd80cac2da48058931")
    record.update(source_commit=summary["source_commit"], steps=STEP, updates=3008,
                  success=summary["policy"]["success"], goal_counts=count.tolist())

def panel(ax, record):
    plot_map(ax, record["maze"])
    data = BASE / record["name"] / f"evaluations/{STEP:09d}_policy.npz"
    plot_rollouts(ax, data, alpha=.45, outcome_colors=True,
                  success_color="#c51b8a", failure_color="#e87924")
    ax.set_title(f"{record['maze'].title()} | seed {record['seed']}\n"
                 f"Success {record['success']:.0%} | goals {sum(c > 0 for c in record['goal_counts'])}/{len(record['goal_counts'])}", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Goal counts: " + str(record["goal_counts"]), fontsize=9)

def render(selected, rows, cols, filename):
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows + .8), squeeze=False)
    for ax, record in zip(axes.flat, selected):
        panel(ax, record)
    fig.suptitle("Corrected DIPO | 200,704 environment transitions | 3,008 learner updates\n"
                 "200 policy rollouts per panel; fresh initial diffusion noise, reverse noise off", fontsize=12, y=.99)
    handles = [Line2D([0], [0], color="#c51b8a", lw=2.5, label="Successful rollout"),
               Line2D([0], [0], color="#e87924", lw=2.5, label="Failed rollout"),
               Line2D([0], [0], color="red", marker="o", lw=0, label="Start")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=.035, right=.98, bottom=.09, top=.86 if rows == 1 else .90,
                        wspace=.16, hspace=.35)
    fig.savefig(BASE / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

render([r for r in records if r["seed"] == 0], 1, 3, "dipo_200k_seed0_simple_medium_hard.png")
render([r for seed in (1, 2) for r in records if r["seed"] == seed], 2, 2,
       "dipo_200k_medium_hard_seed1_seed2.png")
manifest = dict(report_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                evaluation="native DIPO: fresh Gaussian initial noise, reverse diffusion noise off",
                all_episodes_plotted=True, episodes_per_panel=200, records=records)
(BASE / "figure-manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({r["name"]: {"success":r["success"], "goals":r["goal_counts"]} for r in records}, indent=2))
print(BASE)
