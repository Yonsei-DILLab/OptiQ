"""Aggregate the five completed no-floor runs; all learning metrics are read unchanged."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "campaign.json").read_text())
revision = manifest["revision"]
steps = [0, 100, 500, 1000, 2500, 5000] + list(range(10000, 100001, 10000))
keys = ["mmd", "sliced_wasserstein2", "mode_coverage", "high_density_fraction",
        "mode_mass_tv", "mode_mass_js", "effective_modes", "histogram_js",
        "mean_log_target", "boundary_fraction", "outside_fraction"]
runs = []
for seed in range(5):
    path = ROOT / f"seed-{seed}"
    cfg = json.loads((path / "config.json").read_text())
    done = json.loads((path / "completed.json").read_text())
    assert done == {"actor_updates": 100000, "completed": True}
    assert cfg["seed"] == seed and cfg["teacher_std_floor"] is None
    assert cfg["n"] == cfg["m"] == cfg["batch"] == 256
    assert cfg["steps"] == 100000 and cfg["eval_samples"] == 10000
    assert cfg["backend"] == "gpu" and cfg["runtime"]["revision"] == revision
    assert cfg["log_std_min"] == -5 and cfg["log_std_max"] == -3.5 and cfg["initial_log_std"] == -4
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert [r["actor_updates"] for r in records] == steps
    assert records[-1]["Q_evaluations"] == 100000 * 256 * 256
    modes = {}
    for mode in ("full_policy", "mu_only"):
        metric = json.loads((path / "final" / f"metrics_{mode}.json").read_text())
        assert metric == records[-1][mode]
        samples = np.load(path / "final" / f"samples_{mode}.npy", allow_pickle=False)
        assert samples.shape == (10000, 2) and np.isfinite(samples).all()
        assert np.abs(samples).max() <= 40.00001
        modes[mode] = {"metrics": metric, "samples": samples}
    runs.append({"seed": seed, "config": cfg, "records": records, "modes": modes})

target = runs[0]["config"]["target"]
assert all(r["config"]["target"] == target for r in runs)
target_values = {k: target[k] for k in ("means", "std", "weights")}
target_hash = hashlib.sha256(json.dumps(target_values, sort_keys=True).encode()).hexdigest()
assert target_hash == "a22a67f6f37d782fd75961e75f5df13c424fda9624245c99efe93862c929e87c"
summary = {"campaign": manifest["campaign"], "revision": revision,
           "seeds": list(range(5)), "actor_updates_per_seed": 100000,
           "settings": manifest["settings"], "target_sha256": target_hash,
           "primary_mode": "full_policy", "aggregation": "mean and sample standard deviation (ddof=1), n=5",
           "coverage_definition": "occupied GMM component centers out of 40; not density local maxima",
           "analysis_runtime": {"numpy": np.__version__, "matplotlib": matplotlib.__version__}}
for mode in ("full_policy", "mu_only"):
    rows = [{"seed": r["seed"], **{k: r["modes"][mode]["metrics"][k] for k in keys}} for r in runs]
    avg = {k: float(np.mean([row[k] for row in rows])) for k in keys}
    std = {k: float(np.std([row[k] for row in rows], ddof=1)) for k in keys}
    summary[mode] = {"per_seed": rows, "mean": avg, "sample_sd": std}
    with (ROOT / f"{mode}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed"] + keys)
        writer.writeheader()
        writer.writerows(rows + [{"seed": "mean", **avg}, {"seed": "sample_sd", **std}])
summary["input_sha256"] = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(ROOT.glob("seed-*/**/*")) if p.is_file()
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
centers = np.asarray(target["means"])
reference = np.load(ROOT / "reference.npy", allow_pickle=False)
assert reference.shape == (10000, 2) and np.abs(reference).max() <= 40
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
panels = [(reference, "Bounded GMM40 reference", "#738096")]
for r in runs:
    metric = r["modes"]["full_policy"]["metrics"]
    panels.append((r["modes"]["full_policy"]["samples"],
                   f"Seed {r['seed']} | coverage {metric['mode_coverage']}/40 | MMD {metric['mmd']:.4f}",
                   "#2563c5"))
for ax, (points, title, color) in zip(axes.flat, panels):
    ax.scatter(points[:, 0], points[:, 1], s=2, alpha=.45, color=color, linewidths=0, rasterized=True)
    ax.scatter(centers[:, 0], centers[:, 1], s=13, marker="+", color="#17202b", linewidths=.65)
    ax.set(xlim=(-40, 40), ylim=(-40, 40), aspect="equal", title=title,
           xticks=[-40, -20, 0, 20, 40], yticks=[-40, -20, 0, 20, 40])
for ax in axes[-1]: ax.set_xlabel("Physical x")
for ax in axes[:, 0]: ax.set_ylabel("Physical y")
fig.suptitle("GMM40 | no proposal floor | N=M=256 | 100K actor updates\nFull-policy samples: 10,000 per seed", fontsize=15)
fig.savefig(ROOT / "samples-five-seeds.png", dpi=180)
fig.savefig(ROOT / "samples-five-seeds.pdf")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
for ax, key, title in zip(axes, ("mmd", "sliced_wasserstein2", "mode_coverage"),
                          ("MMD (lower is better)", "Sliced W2 (lower is better)", "Component coverage (out of 40)")):
    values = np.array([[record["full_policy"][key] for record in r["records"]] for r in runs])
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    x = np.array(steps) / 1000
    for series in values: ax.plot(x, series, color="#2563c5", alpha=.18, linewidth=.7)
    ax.fill_between(x, mean-sd, mean+sd, color="#2563c5", alpha=.18, label="+/- 1 sample SD")
    ax.plot(x, mean, color="#174aa3", linewidth=2, label="Mean of 5 seeds")
    ax.set(xlabel="Actor updates (thousands)", title=title, xlim=(0, 100))
    ax.grid(axis="y", color="#e3e7ed", linewidth=.6)
    if key == "mode_coverage":
        ax.axhline(40, color="#8993a2", linestyle="--", linewidth=.7)
        ax.set_ylim(0, 41)
    else: ax.set_ylim(bottom=0)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("GMM40 | no proposal floor | full policy | seeds 0-4", fontsize=14)
fig.savefig(ROOT / "learning-curves.png", dpi=180)
fig.savefig(ROOT / "learning-curves.pdf")
plt.close(fig)
manifest["state"] = "completed"
manifest["completion_observed_utc"] = datetime.now(timezone.utc).isoformat()
for job in manifest["jobs"]:
    job["completed"] = True
    job["actor_updates"] = 100000
    job["final_checkpoint"] = job["output"] + "/checkpoints/step_0100000.bin"
(ROOT / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({mode: {"mean": summary[mode]["mean"], "sample_sd": summary[mode]["sample_sd"]}
                  for mode in ("full_policy", "mu_only")}, indent=2))

