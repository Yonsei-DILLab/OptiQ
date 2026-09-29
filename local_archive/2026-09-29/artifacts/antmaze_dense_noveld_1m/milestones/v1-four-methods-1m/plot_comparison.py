"""Compare verified completed v1 policies without pooling training seeds."""
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
base = out.parents[1]
read = lambda p: json.loads(p.read_text())
colors = {"G1/upper": "#1686a6", "G1/lower": "#d6802d", "failure": "#a6a6a6"}
fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.8))
initial = None
geometry = None
verified = {}
for col, method in enumerate(("optiq", "sac", "mfpo", "meow")):
    run = base / "runs" / f"v1-{method}-s0"
    proof = read(run / "archive-verification.json")
    assert proof["passed"] and proof["steps"] == 1000000
    c, r = read(run / "config.json"), read(run / "result.json")
    assert c["seed"] == 0 and r["steps"] == 1000000
    g = c["environment"]
    if geometry is None:
        geometry = g
    for key in ("walls", "goals", "horizon"):
        assert g[key] == geometry[key]
    for row, mode in enumerate(("policy", "native")):
        ax = axes[row, col]
        label = mode + "-fixed"
        f = run / "rollouts" / f"1000000-{label}.npz"
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        assert sha == proof["sha256"][str(f.relative_to(run))]
        z = np.load(f, allow_pickle=False)
        states = z["initial_simulator_state"]
        np.testing.assert_array_equal(states, np.broadcast_to(states[0], states.shape))
        if initial is None:
            initial = states[0]
        np.testing.assert_array_equal(initial, states[0])
        s = r["summaries"][label]
        assert s["episodes"] == 100
        for x, y in g["walls"]:
            ax.add_patch(Rectangle((x-2, y-2), 4, 4, color="#42484e", zorder=1))
        for xy, n, route in zip(z["xy"], z["lengths"], s["routes"]):
            line = xy[:int(n)+1]
            ax.plot(line[:, 0], line[:, 1], color=colors[route], lw=.8,
                    alpha=.22 if route != "failure" else .7, zorder=3)
        ax.scatter(*z["xy"][0, 0], marker="*", c="black", s=70, zorder=6)
        for x, y in g["goals"]:
            ax.add_patch(Circle((x, y), .5, color="#44a25f", zorder=5))
            ax.text(x, y+.85, "Goal", ha="center", fontsize=8, zorder=6)
        walls = np.asarray(g["walls"])
        ax.set(xlim=(walls[:, 0].min()-2, walls[:, 0].max()+2),
               ylim=(walls[:, 1].min()-2, walls[:, 1].max()+2),
               aspect="equal", xlabel="x", ylabel="y")
        if row == 0:
            ax.set_xlabel("")
        counts = s["successful_routes"]["counts"]
        up, low = counts.get("G1/upper", 0), counts.get("G1/lower", 0)
        fail = 100-up-low
        name = {"optiq": "OptiQ", "sac": "SAC", "mfpo": "MFPO", "meow": "MEOW"}[method]
        meaning = ("random z + sigma" if method == "optiq" else "direct policy draw") if mode == "policy" else (
            {"optiq": "random z, mu-only", "sac": "tanh(mu)", "mfpo": "Q-best-of-10", "meow": "center prior"}[method])
        ax.set_title(f"{name} · {meaning}\nSuccess {up+low}/100 · upper {up}, lower {low}, fail {fail}", fontsize=10)
        verified[f"{method}-{label}"] = dict(successes=up+low, upper=up, lower=low,
            failures=fail, raw_sha256=sha, identical_initial_state_across_all_panels=True)
fig.suptitle("AntMaze v1 · completed 1M policies · training seed 0\nTop: direct stochastic policy | Bottom: native evaluation", fontsize=14)
fig.legend([Line2D([0], [0], color=colors[k], lw=2) for k in colors],
           ["Upper route", "Lower route", "Failure"], loc="lower center",
           bbox_to_anchor=(.5, .045), ncol=3, frameon=False)
fig.text(.5, .030, "100 rollouts per panel from the same full simulator state. No extra DACER action noise or NovelD reward in evaluation.", ha="center", fontsize=8)
fig.text(.5, .010, "Dense + NovelD training; native model/optimizer settings. One seed per method; v1 comparison complete. Remaining maps are still in progress.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, .09, 1, .92), h_pad=3.0)
for ext in ("png", "pdf"):
    fig.savefig(out / f"fixed-state-comparison.{ext}", dpi=180)
plt.close(fig)
(out / "fixed-state-verification.json").write_text(json.dumps(verified, indent=2)+"\n")
print(json.dumps(verified, indent=2))
