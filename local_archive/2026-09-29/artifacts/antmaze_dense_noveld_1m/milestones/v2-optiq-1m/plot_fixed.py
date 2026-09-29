"""Plot one verified production policy, without pooling reset conditions."""
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
run = out.parents[1] / "runs" / "v2-optiq-s0"
read = lambda p: json.loads(p.read_text())
proof = read(run / "archive-verification.json")
assert proof["passed"] and proof["steps"] == 1000000
config, result = read(run / "config.json"), read(run / "result.json")
assert config["seed"] == 0 and result["updates"] == 995000
assert result["task"] == "v2" and result["method"] == "optiq"
g = config["environment"]
colors = {"G1": "#1686a6", "G2": "#d6802d", "failure": "#979da3"}
modes = [("native", "Random z · mu-only"),
         ("policy", "Random z + conditional sigma"),
         ("zero_z", "Zero z · mu-only")]
fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.1))
initial = None
verified = {}
for ax, (mode, title) in zip(axes, modes):
    label = mode + "-fixed"
    file = run / "rollouts" / ("1000000-" + label + ".npz")
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    assert sha == proof["sha256"][str(file.relative_to(run))]
    with np.load(file, allow_pickle=False) as z:
        states = z["initial_simulator_state"]
        np.testing.assert_array_equal(states, np.broadcast_to(states[0], states.shape))
        if initial is None:
            initial = states[0].copy()
        np.testing.assert_array_equal(initial, states[0])
        s = result["summaries"][label]
        assert s["episodes"] == 100 and s["reset_mode"] == "identical_full_state"
        for x, y in g["walls"]:
            ax.add_patch(Rectangle((x-2, y-2), 4, 4, color="#42484e", zorder=1))
        for xy, n, route in zip(z["xy"], z["lengths"], s["routes"]):
            line = xy[:int(n)+1]
            key = route.split("/")[0] if route != "failure" else route
            ax.plot(line[:, 0], line[:, 1], color=colors[key],
                    alpha=.24 if key != "failure" else .85,
                    lw=.85 if key != "failure" else 1.3, zorder=3)
            if key == "failure":
                ax.scatter(*line[-1], marker="x", color=colors[key], s=22, zorder=6)
        ax.scatter(*z["xy"][0, 0], marker="*", c="black", s=85, zorder=6)
    for i, (x, y) in enumerate(g["goals"], 1):
        ax.add_patch(Circle((x, y), .5, color="#44a25f", zorder=5))
        ax.text(x, y+1.2, f"G{i}", ha="center", fontsize=10, zorder=6)
    walls = np.asarray(g["walls"])
    ax.set(xlim=(walls[:, 0].min()-2, walls[:, 0].max()+2),
           ylim=(walls[:, 1].min()-2, walls[:, 1].max()+2), aspect="equal",
           xlabel="x", ylabel="y")
    goals = s["successful_goals"]["counts"]
    g1, g2 = goals.get("1", 0), goals.get("2", 0)
    fail = 100-g1-g2
    ax.set_title(f"{title}\nSuccess {g1+g2}/100 · G1 {g1}, G2 {g2}, fail {fail}", fontsize=10)
    verified[label] = dict(successes=g1+g2, goals=goals, failures=fail,
        routes=s["successful_routes"]["counts"], raw_sha256=sha,
        identical_state_across_modes=True)
fig.suptitle("AntMaze v2 · OptiQ · 1M interactions · training seed 0", fontsize=15)
fig.legend([Line2D([0], [0], color=colors[k], lw=2) for k in colors],
           ["Successful path to G1", "Successful path to G2", "Failure"],
           loc="lower center", bbox_to_anchor=(.5, .080), ncol=3, frameon=False)
fig.text(.5, .054, "100 rollouts per panel, same full initial simulator state. Random z is resampled at every action.",
         ha="center", fontsize=9)
fig.text(.5, .022, "Dense + NovelD training; no added DACER noise or NovelD reward in evaluation. Single training seed.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .21, 1, .93))
for ext in ("png", "pdf"):
    fig.savefig(out / f"fixed-state-trajectories.{ext}", dpi=180)
plt.close(fig)
(out / "fixed-state-verification.json").write_text(json.dumps(dict(
    source_commit=result["source_commit"], training_seed=0, steps=1000000,
    archive_verified=True, evaluations=verified,
    plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+"\n")
print(json.dumps(verified, indent=2))
