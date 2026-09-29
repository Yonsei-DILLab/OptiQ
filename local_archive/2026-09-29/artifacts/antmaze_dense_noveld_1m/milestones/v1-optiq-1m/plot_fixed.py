"""Render verified production rollouts from one completed policy."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle

out = Path(__file__).resolve().parent
run = out.parents[1] / "runs" / "v1-optiq-s0"
read = lambda p: json.loads(p.read_text())
proof = read(run / "archive-verification.json")
assert proof["passed"] and proof["steps"] == 1000000
config, result = read(run / "config.json"), read(run / "result.json")
assert config["seed"] == 0 and result["updates"] == 995000
g = config["environment"]
colors = {"G1/upper": "#1686a6", "G1/lower": "#d6802d", "failure": "#a6a6a6"}
modes = [("native", "Random z · mu-only"),
         ("policy", "Random z + conditional sigma"),
         ("zero_z", "Zero z · mu-only")]
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))
initial = None
verified = {}
for ax, (mode, title) in zip(axes, modes):
    label = mode + "-fixed"
    file = run / "rollouts" / ("1000000-" + label + ".npz")
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    assert sha == proof["sha256"][str(file.relative_to(run))]
    z = np.load(file, allow_pickle=False)
    states = z["initial_simulator_state"]
    np.testing.assert_array_equal(states, np.broadcast_to(states[0], states.shape))
    if initial is None:
        initial = states[0]
    np.testing.assert_array_equal(initial, states[0])
    s = result["summaries"][label]
    assert s["episodes"] == 100 and s["reset_mode"] == "identical_full_state"
    for x, y in g["walls"]:
        ax.add_patch(Rectangle((x-2, y-2), 4, 4, color="#42484e", zorder=1))
    for xy, n, route in zip(z["xy"], z["lengths"], s["routes"]):
        line = xy[:int(n)+1]
        ax.plot(line[:, 0], line[:, 1], color=colors[route],
                alpha=.23 if route != "failure" else .65, lw=.9, zorder=3)
    start = z["xy"][0, 0]
    ax.scatter(*start, marker="*", c="black", s=85, zorder=6)
    for x, y in g["goals"]:
        ax.add_patch(Circle((x, y), .5, color="#44a25f", zorder=5))
        ax.text(x, y+.85, "Goal", ha="center", fontsize=9, zorder=6)
    walls = np.asarray(g["walls"])
    ax.set(xlim=(walls[:, 0].min()-2, walls[:, 0].max()+2),
           ylim=(walls[:, 1].min()-2, walls[:, 1].max()+2), aspect="equal",
           xlabel="x", ylabel="y")
    counts = s["successful_routes"]["counts"]
    upper, lower = counts.get("G1/upper", 0), counts.get("G1/lower", 0)
    fail = 100-upper-lower
    ax.set_title(f"{title}\nSuccess {upper+lower}/100 · upper {upper}, lower {lower}, fail {fail}", fontsize=10)
    verified[label] = dict(successes=upper+lower, upper=upper, lower=lower, failures=fail,
                           raw_sha256=sha, identical_state_across_modes=True)
fig.suptitle("AntMaze v1 · OptiQ · 1M interactions · training seed 0", fontsize=15)
fig.legend([Line2D([0], [0], color=colors[k], lw=2) for k in colors],
           ["Upper route", "Lower route", "Failure"], loc="lower center",
           bbox_to_anchor=(.5, .064), ncol=3, frameon=False)
fig.text(.5, .047, "Same full initial simulator state in all 300 rollouts; one trained policy; random z is resampled at each action.",
         ha="center", fontsize=9)
fig.text(.5, .012, "Dense + NovelD during training. No added DACER action noise or NovelD reward during evaluation. Other 15 runs unfinished.",
         ha="center", fontsize=8.5)
fig.tight_layout(rect=(0, .19, 1, .92))
for ext in ("png", "pdf"):
    fig.savefig(out / f"fixed-state-trajectories.{ext}", dpi=180)
plt.close(fig)
(out / "fixed-state-verification.json").write_text(json.dumps(dict(
    source_commit=result["source_commit"], training_seed=0, steps=1000000,
    archive_verified=True, evaluations=verified), indent=2)+"\n")
print(json.dumps(verified, indent=2))
