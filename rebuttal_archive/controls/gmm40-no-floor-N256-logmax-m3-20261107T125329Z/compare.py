"""Paired, five-seed comparison of the two no-floor log-scale bounds."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT/"campaign.json").read_text())
BASE = ROOT.parent/manifest["comparison"]["baseline_campaign"]
old = json.loads((BASE/"summary.json").read_text())
new = json.loads((ROOT/"summary.json").read_text())
assert old["seeds"] == new["seeds"] == list(range(5))
assert all(j["initial_checkpoint_matches_baseline"] for j in manifest["jobs"])
assert np.array_equal(np.load(BASE/"reference.npy",allow_pickle=False),
                      np.load(ROOT/"reference.npy",allow_pickle=False))
keys = ["mmd", "sliced_wasserstein2", "mode_coverage", "high_density_fraction",
        "mode_mass_tv", "mode_mass_js", "effective_modes", "histogram_js"]
comparison = {
    "baseline_campaign": old["campaign"], "new_campaign": new["campaign"],
    "changed_parameter": "log_std_max", "baseline_value": -3.5, "new_value": -3.0,
    "proposal_floor": None, "seeds": list(range(5)), "updates_per_seed": 100000,
    "paired_initialization_verified": True,
    "reference_array_identical": True,
    "initial_checkpoint_sha256": {str(j["seed"]):j["initial_checkpoint_sha256"] for j in manifest["jobs"]},
    "statistics": "mean and sample SD (ddof=1); paired difference is new minus baseline",
}
history = {"baseline": [], "new": []}
sigma = {"baseline": [], "new": []}
for seed in range(5):
    configs = [json.loads((p/f"seed-{seed}/config.json").read_text()) for p in (BASE,ROOT)]
    assert configs[0].keys() == configs[1].keys()
    differences = {k for k in configs[0] if configs[0][k] != configs[1][k]}
    assert differences == {"log_std_max", "runtime"}, differences
    assert configs[0]["runtime"]["packages"] == configs[1]["runtime"]["packages"]
    for label, p in (("baseline", BASE), ("new", ROOT)):
        records = [json.loads(line) for line in (p/f"seed-{seed}/metrics.jsonl").read_text().splitlines()]
        history[label].append(records)
        sigma[label].append({key: records[-1]["training"][key]
                             for key in ("sigma_mean","sigma_min","sigma_max")})
comparison["conditional_sigma_final_training_window"] = sigma
for mode in ("full_policy", "mu_only"):
    left, right = old[mode], new[mode]
    delta = [{"seed": a["seed"], **{k: b[k]-a[k] for k in keys}}
             for a,b in zip(left["per_seed"],right["per_seed"])]
    comparison[mode] = {
        "baseline_mean": left["mean"], "baseline_sample_sd": left["sample_sd"],
        "new_mean": right["mean"], "new_sample_sd": right["sample_sd"],
        "paired_difference": {
            "per_seed": delta,
            "mean": {k: float(np.mean([r[k] for r in delta])) for k in keys},
            "sample_sd": {k: float(np.std([r[k] for r in delta],ddof=1)) for k in keys},
        },
    }
    with (ROOT/f"comparison-{mode}.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["profile","statistic"]+keys)
        writer.writeheader()
        for name, block in (("[-5,-3.5]",left),("[-5,-3.0]",right)):
            for stat in ("mean","sample_sd"):
                writer.writerow({"profile":name,"statistic":stat,**{k:block[stat][k] for k in keys}})
(ROOT/"comparison.json").write_text(json.dumps(comparison,indent=2,allow_nan=False)+"\n")

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
settings=[("mmd","Full-policy MMD (lower is better)"),
          ("sliced_wasserstein2","Full-policy sliced W2 (lower is better)"),
          ("mode_coverage","Component coverage (out of 40)"),
          ("sigma_mean","Conditional sigma (training mean)")]
colors={"baseline":"#2563c5","new":"#d46b16"}
labels={"baseline":"log-sigma [-5, -3.5]","new":"log-sigma [-5, -3.0]"}
for ax,(key,title) in zip(axes.flat,settings):
    for label in ("baseline","new"):
        histories=history[label]
        records=histories[0]
        idx=[i for i,r in enumerate(records) if key!="sigma_mean" or r["training"]]
        x=np.array([records[i]["actor_updates"] for i in idx])/1000
        values=np.array([[r[i]["training"][key] if key=="sigma_mean" else r[i]["full_policy"][key]
                          for i in idx] for r in histories])
        mean,sd=values.mean(0),values.std(0,ddof=1)
        ax.plot(x,mean,color=colors[label],linewidth=2,label=labels[label])
        ax.fill_between(x,mean-sd,mean+sd,color=colors[label],alpha=.15)
        if key=="sigma_mean":
            bound=-3.5 if label=="baseline" else -3.0
            ax.axhline(np.exp(bound),color=colors[label],linestyle="--",linewidth=.8,alpha=.7)
    ax.set(xlabel="Actor updates (thousands)",title=title,xlim=(0,100))
    ax.grid(axis="y",color="#e3e7ed",linewidth=.6)
    ax.set_ylim(bottom=0)
    if key=="mode_coverage": ax.set_ylim(0,41)
axes[0,0].legend(fontsize=9,frameon=False)
fig.suptitle("GMM40 | no proposal floor | matched seeds 0-4\nMean +/- 1 sample SD; N=M=256, batch=256, 100K updates",fontsize=14)
fig.savefig(ROOT/"range-comparison.png",dpi=180)
fig.savefig(ROOT/"range-comparison.pdf")
plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
palette = plt.get_cmap("tab10")
for ax, key, title in zip(axes, ("mmd", "sliced_wasserstein2", "mode_coverage"),
                          ("Full-policy MMD", "Full-policy sliced W2", "Component coverage / 40")):
    a = np.array([r[key] for r in old["full_policy"]["per_seed"]])
    b = np.array([r[key] for r in new["full_policy"]["per_seed"]])
    for seed, (left, right) in enumerate(zip(a, b)):
        jitter = (seed-2)*.035
        ax.plot([jitter, 1+jitter], [left, right], color=palette(seed), alpha=.45, linewidth=.8)
        ax.scatter([jitter, 1+jitter], [left, right], color=palette(seed), s=28, zorder=3)
    ax.errorbar([0,1], [a.mean(),b.mean()], yerr=[a.std(ddof=1),b.std(ddof=1)],
                fmt="D", color="black", markerfacecolor="white", markersize=6,
                capsize=5, linewidth=1.5, zorder=5, label="Mean +/- sample SD")
    ax.set(xticks=[0,1],xticklabels=["max -3.5","max -3.0"],xlim=(-.25,1.25),title=title)
    ax.grid(axis="y",color="#e3e7ed",linewidth=.6)
    if key == "mode_coverage": ax.set_ylim(38.5,40.5)
    else: ax.set_ylim(bottom=0)
axes[0].legend(fontsize=8,frameon=False)
fig.suptitle("GMM40 at 100K updates | no proposal floor | matched seeds 0-4",fontsize=13)
fig.savefig(ROOT/"final-metrics-comparison.png",dpi=180)
fig.savefig(ROOT/"final-metrics-comparison.pdf")
plt.close(fig)
print(json.dumps({m:comparison[m] for m in ("full_policy","mu_only")},indent=2))

