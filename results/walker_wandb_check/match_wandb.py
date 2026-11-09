"""(1) Plot W&B paper-era best-of-N curves (zero_z, stochastic_z) for 5 MuJoCo tasks.
(2) From our 3-mode runs (original seeds 0-9 + duplicate seeds 0-4), pick 5 runs with distinct
    seeds whose mean zero_z curve is closest (RMSE over the whole curve) to the W&B zero_z curve,
    and report last-100K means for zero_z / stochastic_z / stochastic_z_sigma on those runs.
Usage: python match_wandb.py <csv_dir> <orig_eval_dir> <dup_eval_dir> <out_dir>
"""
import glob, itertools, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV, ORIG, DUP, OUT = sys.argv[1:5]
TASKS = [("hopper", "Hopper-v4"), ("walker2d", "Walker2d-v4"), ("halfcheetah", "HalfCheetah-v4"),
         ("ant", "Ant-v4"), ("humanoid", "Humanoid-v4")]
MODES = ["zero_z", "stochastic_z", "stochastic_z_sigma"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RED, BLUE, ORANGE, GREEN = "#d6334a", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

def load(root, task, tag):
    out = {}
    for d in glob.glob(os.path.join(root, f"{task}-ibolt-*")):
        s = int(re.search(r"-s(\d+)_", os.path.basename(d)).group(1))
        if tag == "dup" and s > 4:
            continue
        r = {}
        for m in MODES:
            z = np.load(glob.glob(os.path.join(d, "**", f"evaluations_{m}.npz"), recursive=True)[0])
            r[m] = (z["timesteps"], z["results"].mean(1))
        out[(s, tag)] = r
    return out

def last100(t, y):
    return y[t > t[-1] - 100_000].mean()

def style(ax, title):
    ax.set_facecolor(SURF); ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Environment steps (M)"); ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True); ax.set_xlim(0, 1.0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig1, ax1 = plt.subplots(1, 5, figsize=(20, 4.2), facecolor=SURF)
fig2, ax2 = plt.subplots(1, 5, figsize=(20, 4.2), facecolor=SURF)
rows = []
for k, (task, title) in enumerate(TASKS):
    W = np.loadtxt(os.path.join(CSV, f"wandb_best_{task}.csv"), delimiter=",", skiprows=1)
    tw = W[:, 0]
    # (1) evidence plot
    a = ax1[k]
    for col, color, ls, lab in [(1, RED, "-", "W&B best-of-N, zero_z"), (3, ORANGE, "--", "W&B best-of-N, stochastic_z")]:
        a.fill_between(tw / 1e6, W[:, col] - W[:, col + 1], W[:, col] + W[:, col + 1], color=color, alpha=0.12, linewidth=0)
        a.plot(tw / 1e6, W[:, col], color=color, linestyle=ls, linewidth=1.7, label=lab)
    style(a, title)
    wz_last = W[tw > tw[-1] - 100_000, 1].mean(); ws_last = W[tw > tw[-1] - 100_000, 3].mean()

    # (2) matching
    pool = {**load(ORIG, task, "orig"), **load(DUP, task, "dup")}
    t = pool[(0, "orig")]["zero_z"][0]
    n = min(len(r["zero_z"][0]) for r in pool.values()); t = t[:n]
    target = np.interp(t, tw, W[:, 1])
    Z = {key: r["zero_z"][1][:n] for key, r in pool.items()}
    options = {s: [key for key in pool if key[0] == s] for s in range(10)}
    best = None
    for seeds in itertools.combinations(range(10), 5):
        for pick in itertools.product(*(options[s] for s in seeds)):
            mu = np.mean([Z[p] for p in pick], axis=0)
            err = np.sqrt(np.mean((mu - target) ** 2))
            if best is None or err < best[0]:
                best = (err, pick)
    err, pick = best
    res = {m: np.array([last100(pool[p][m][0][:n], pool[p][m][1][:n]) for p in pick]) for m in MODES}
    allz = np.array([last100(t, Z[p]) for p in pool])
    rows.append((task, pick, err, wz_last, ws_last, res, allz.mean()))

    a = ax2[k]
    a.fill_between(tw / 1e6, W[:, 1] - W[:, 2], W[:, 1] + W[:, 2], color=RED, alpha=0.10, linewidth=0)
    a.plot(tw / 1e6, W[:, 1], color=RED, linewidth=1.7, label="W&B best-of-N, zero_z (paper)")
    for m, color, ls, lab in [("zero_z", BLUE, "--", "ours (matched 5), zero_z"),
                              ("stochastic_z_sigma", GREEN, ":", "ours (matched 5), z + σ")]:
        Y = np.stack([pool[p][m][1][:n] for p in pick]); mu, sd = Y.mean(0), Y.std(0)
        a.fill_between(t / 1e6, mu - sd, mu + sd, color=color, alpha=0.10, linewidth=0)
        a.plot(t / 1e6, mu, color=color, linestyle=ls, linewidth=1.7, label=lab)
    style(a, title)

for fig, axes, name, note in [
        (fig1, ax1, "wandb_best_curves.png", "W&B runs (OptiQ/NM-abal + OptiQ/gmm-trg), paper temperature, DACER on, N=M=64; best run per seed (s0-4) by last-100K zero_z; mean ± 1 std"),
        (fig2, ax2, "matched_runs.png", "Ours: 5 runs (distinct seeds, from s0-9 + duplicate s0-4) whose mean zero_z curve is closest (RMSE) to the W&B curve; mean ± 1 std")]:
    axes[0].set_ylabel("Evaluation return")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.995, 0.01, note, ha="right", va="bottom", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93)); fig.savefig(os.path.join(OUT, name), dpi=150, facecolor=SURF)

print(f"{'task':12s} {'W&B zero':>9s} {'W&B stoch':>9s} | {'ours zero':>13s} {'ours stoch':>13s} {'ours z+σ':>13s} | RMSE  picked runs   (all-runs zero mean)")
for task, pick, err, wz, ws, res, allm in rows:
    f = lambda a: f"{a.mean():6.0f}±{a.std():4.0f}"
    print(f"{task:12s} {wz:9.0f} {ws:9.0f} | {f(res['zero_z']):>13s} {f(res['stochastic_z']):>13s} {f(res['stochastic_z_sigma']):>13s} | {err:4.0f}  "
          + " ".join(f"s{s}{'d' if g == 'dup' else ''}" for s, g in sorted(pick)) + f"   ({allm:.0f})")
