"""Render v3 comparisons only after all four 1M archives pass verification."""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

out = Path(__file__).resolve().parent
base = out.parents[1]
methods = ("optiq", "sac", "mfpo", "meow")
names = dict(optiq="OptiQ", sac="SAC", mfpo="MFPO", meow="MEOW")
native = dict(optiq="random z, mu-only", sac="tanh(mu)",
              mfpo="Q-best-of-10", meow="center prior")
colors = {"G1": "#1686a6", "G2": "#d6802d", "failure": "#a6a6a6"}
read = lambda path: json.loads(path.read_text())
runs = {}
initial = None
geometry = None
verified = {}

# Check all prerequisites before writing either figure.
for method in methods:
    run = base / "runs" / f"v3-{method}-s0"
    proof = read(run / "archive-verification.json")
    config, result = read(run / "config.json"), read(run / "result.json")
    assert proof["passed"] and proof["steps"] == result["steps"] == 1000000
    assert config["seed"] == 0
    env = config["environment"]
    if geometry is None:
        geometry = env
    for key in ("walls", "goals", "horizon"):
        assert env[key] == geometry[key]
    runs[method] = {}
    for mode in ("policy", "native"):
        label = mode + "-fixed"
        path = run / "rollouts" / f"1000000-{label}.npz"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == proof["sha256"][str(path.relative_to(run))]
        with np.load(path, allow_pickle=False) as data:
            z = {key: data[key] for key in ("initial_simulator_state", "xy", "lengths")}
        states = z["initial_simulator_state"]
        np.testing.assert_array_equal(states, np.broadcast_to(states[0], states.shape))
        if initial is None:
            initial = states[0]
        np.testing.assert_array_equal(initial, states[0])
        summary = result["summaries"][label]
        assert summary["episodes"] == len(z["lengths"]) == len(summary["routes"]) == 100
        counts = summary["successful_goals"]["counts"]
        g1, g2 = counts.get("1", 0), counts.get("2", 0)
        verified[f"{method}-{label}"] = dict(
            successes=g1+g2, goal1=g1, goal2=g2, failures=100-g1-g2,
            successful_routes=summary["successful_routes"]["counts"],
            raw_sha256=sha, identical_initial_state_across_all_panels=True)
        runs[method][mode] = (z, summary)

for mode in ("policy", "native"):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 12.3))
    for ax, method in zip(axes.flat, methods):
        z, summary = runs[method][mode]
        for x, y in geometry["walls"]:
            ax.add_patch(Rectangle((x-2, y-2), 4, 4, color="#42484e", zorder=1))
        for xy, length, route in zip(z["xy"], z["lengths"], summary["routes"]):
            line = xy[:int(length)+1]
            key = "failure" if route == "failure" else route.split("/")[0]
            ax.plot(line[:, 0], line[:, 1], color=colors[key], lw=.8,
                    alpha=.22 if key != "failure" else .6, zorder=3)
        ax.scatter(*z["xy"][0, 0], marker="*", c="black", s=70, zorder=6)
        for goal_index, (x, y) in enumerate(geometry["goals"], 1):
            ax.add_patch(Circle((x, y), .5, color="#44a25f", zorder=5))
            ax.text(x, y+.85, f"G{goal_index}", ha="center", fontsize=9, zorder=6)
        walls = np.asarray(geometry["walls"])
        ax.set(xlim=(walls[:, 0].min()-2, walls[:, 0].max()+2),
               ylim=(walls[:, 1].min()-2, walls[:, 1].max()+2),
               aspect="equal", xlabel="x", ylabel="y")
        counts = verified[f"{method}-{mode}-fixed"]
        meaning = ("random z + sigma" if method == "optiq" else "direct policy draw") if mode == "policy" else native[method]
        ax.set_title(f"{names[method]} · {meaning}\nSuccess {counts['successes']}/100 · G1 {counts['goal1']}, G2 {counts['goal2']}, fail {counts['failures']}", fontsize=10)
    title = "Direct stochastic policy draws" if mode == "policy" else "Native evaluation (auxiliary comparison)"
    fig.suptitle(f"AntMaze v3 · completed 1M policies · training seed 0\n{title}; identical full initial state", fontsize=14)
    fig.legend([Line2D([0], [0], color=colors[key], lw=2) for key in colors],
               ["Successful path to G1", "Successful path to G2", "Failure"],
               loc="lower center", bbox_to_anchor=(.5, .05), ncol=3, frameon=False)
    fig.text(.5, .035, "100 rollouts per panel. No extra DACER action noise or NovelD reward in evaluation.", ha="center", fontsize=8)
    fig.text(.5, .016, "Dense + NovelD training; native model/optimizer settings. One training seed per method.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .925), h_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{mode}-fixed-comparison.{ext}", dpi=180)
    plt.close(fig)

(out / "fixed-state-verification.json").write_text(json.dumps(verified, indent=2)+"\n")
print(json.dumps(verified, indent=2))
