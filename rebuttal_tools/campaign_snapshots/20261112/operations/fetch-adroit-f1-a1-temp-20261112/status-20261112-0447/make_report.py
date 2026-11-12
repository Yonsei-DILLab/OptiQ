"""Exploratory pilot-vs-baseline report; never selects a temperature or a run."""
from pathlib import Path
import collections
import datetime
import json
import math
import shutil
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parent
if not (P / "snapshot.json").exists():
    shutil.copy2(P.parent / "snapshot.json", P / "snapshot.json")
D = json.loads((P / "snapshot.json").read_text())
TEMPS = [.05, .1, .25, .5, 1.]
COLORS = dict(zip(TEMPS, ["#0072B2", "#D55E00", "#009E73", "#AA4499", "#9A7900"]))
TASKS = {"fetch": ["reach", "push", "pick-and-place", "slide"],
         "adroit": ["door", "hammer", "pen", "relocate"]}
STAMP = datetime.datetime.fromtimestamp(D["at"], datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def aggregate(rows, required=None):
    rows = sorted(rows, key=lambda r: r["job"]["seed"])
    if required is not None and [r["job"]["seed"] for r in rows] != required:
        return None
    if len(rows) < 2:
        return None
    maps = [{e["steps"]: e for e in r["curve"] if e["episodes"] == 10} for r in rows]
    grid = sorted(set.intersection(*(set(m) for m in maps)))
    if not grid:
        return None
    for step in grid:
        assert len({m[step]["updates"] for m in maps}) == 1
        assert len({m[step]["primary_mode"] for m in maps}) == 1
    result = dict(ids=[r["job"]["id"] for r in rows], seeds=[r["job"]["seed"] for r in rows],
                  steps=grid, maps=maps, updates=[maps[0][s]["updates"] for s in grid])
    result["success"] = np.array([[m[s]["success"] * 100 for s in grid] for m in maps])
    result["return"] = np.array([[m[s]["return_"] for s in grid] for m in maps])
    assert np.isfinite(result["success"]).all() and np.isfinite(result["return"]).all()
    return result


groups = {}
checks = []
for family, tasks in TASKS.items():
    for task in tasks:
        rows = [r for r in D["rows"] if r["job"]["family"] == family and r["job"]["task"] == task]
        configured = [r for r in rows if r["config"]]
        common_keys = ["warmup", "batch_size", "utd", "num_envs"]
        if family == "adroit":
            common_keys += ["actor_hidden", "critic_hidden", "critic_layer_norm", "horizon",
                            "demonstration_ratio", "offline_pretraining_updates",
                            "demonstration_dataset_sha256", "normalize_observations", "normalize_rewards"]
        else:
            common_keys += ["HER", "normalize_observations", "normalization_recipe"]
        for key in common_keys:
            assert len({json.dumps(r["config"][key], sort_keys=True) for r in configured}) == 1, (family, task, key)
        checks.append(dict(family=family, task=task, shared_configuration_keys=common_keys))
        sac_rows = [r for r in rows if r["job"]["method"] == "sac" and r["curve"]]
        gg = {"sac": aggregate(sac_rows)}
        for t in TEMPS:
            gg[t] = aggregate([r for r in rows if r["job"]["phase"] == "pilot" and r["job"]["temperature"] == t], [5, 6])
        groups[(family, task)] = gg


tables = []
for family, tasks in TASKS.items():
    for metric in ["success", "score"]:
        fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4))
        for ax, task in zip(axes.flat, tasks):
            gg = groups[(family, task)]
            ends = []
            sac = gg["sac"]
            for key in [*TEMPS, "sac"]:
                g = gg[key]
                if g is None:
                    continue
                x = np.array(g["steps"]) / 1e6
                vals = g["success"] if metric == "success" else (
                    (g["return"] + 50) / 50 * 100 if family == "fetch" else (g["return"] + 20) / 2020 * 100)
                mean, sd = vals.mean(0), vals.std(0, ddof=1)
                color = "#161C24" if key == "sac" else COLORS[key]
                label = (f"JAX SAC (n={len(g['seeds'])})" if key == "sac" else f"iBOLT T={key:g} (n=2)")
                ax.plot(x, mean, color=color, lw=2.3 if key == "sac" else 1.65,
                        ls="--" if key == "sac" and len(g["seeds"]) < 5 else "-", label=label)
                ax.fill_between(x, np.maximum(0, mean-sd), np.minimum(100, mean+sd), color=color, alpha=.09)
                ends.append(x[-1])
            end = max(.6, math.ceil(max(ends, default=.6)*2)/2)
            if sac:
                end_sac = sac["steps"][-1]/1e6
                ax.axvspan(end_sac, end, color="#888888", alpha=.045, zorder=0)
                ax.axvline(end_sac, ls=":", color="#555555", lw=1, alpha=.6)
                ax.text(.985, .04, "SAC seeds: " + ",".join(map(str, sac["seeds"])) +
                        f"\nCohort observed through {end_sac:.2f}M", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8, color="#444444",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
            missing = [f"{t:g}" for t in TEMPS if gg[t] is None]
            ax.text(.02, .97, "Awaiting both pilot seeds: T=" + ", ".join(missing) if missing else "All temperatures have paired pilot evaluations",
                    transform=ax.transAxes, va="top", fontsize=8, color="#555555",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
            ax.set_title(task.replace("-", " ").title() + (" (20 cm)" if family == "fetch" and task == "slide" else ""), loc="left", fontweight="bold")
            ax.set_xlim(0, end)
            ax.set_ylim(-2, 109)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(alpha=.17)
            ax.set_xlabel("Environment steps (M)")
            ax.set_ylabel("Success rate (%)" if metric == "success" else "Reward-range score (not D4RL)")
            ax.legend(loc="center right", fontsize=8, framealpha=.9)
        title_metric = "success rate" if metric == "success" else "derived reward-range score"
        fig.suptitle(f"{family.title()} | {title_metric} | exploratory comparison", x=.065, ha="left", fontsize=17, fontweight="bold", y=.98)
        settings = ("F1: 256 x 2 | LayerNorm off | HER | stochastic z, conditional mean" if family == "fetch" else
                    "A1: 256 x 2 | critic LayerNorm | fixed expert25 | BC=0 | z=0, conditional mean")
        fig.text(.065, .925, settings + "\nColored: iBOLT pilot seeds 5,6. Black: available JAX SAC seeds 0-4. Different seed cohorts; no final superiority claim.", fontsize=9.6, color="#444444", va="top")
        foot = "Mean +/- sample SD, 10 episodes/run/evaluation. Exact shared steps within each cohort; no smoothing, interpolation, or future projection."
        if metric == "score":
            foot += "\nScore = 100(R+50)/50 for Fetch; 100(R+20)/2020 for Adroit. Derived from raw return; not official normalized/D4RL score."
        else:
            foot += "\nDotted line: latest SAC cohort step. Later iBOLT observations have no equally mature complete SAC cohort yet."
        fig.text(.065, .024, STAMP + " | " + foot, fontsize=8.6, color="#555555")
        fig.subplots_adjust(left=.075, right=.985, bottom=.14, top=.81, hspace=.44, wspace=.21)
        fig.savefig(P / f"{family}-{metric}.png", dpi=170)
        fig.savefig(P / f"{family}-{metric}.pdf")
        plt.close(fig)

for (family, task), gg in groups.items():
    anchor = 1003520 if family == "adroit" else 1000192
    sac = gg["sac"]
    for key, g in gg.items():
        if g is None or anchor not in g["steps"]:
            continue
        idx = g["steps"].index(anchor)
        tail = g["success"][:, max(0, idx-4):idx+1].mean(1)
        value = g["success"][:, idx]
        ret = g["return"][:, idx]
        tables.append(dict(family=family, task=task, method="JAX SAC" if key == "sac" else "iBOLT",
                           temperature=None if key == "sac" else key, seeds=g["seeds"], step=anchor,
                           success=value.tolist(), mean=float(value.mean()), sd=float(value.std(ddof=1)),
                           previous_five_steps=g["steps"][max(0, idx-4):idx+1],
                           previous_five_mean=float(tail.mean()), previous_five_sd=float(tail.std(ddof=1)),
                           raw_return_mean=float(ret.mean()), raw_return_sd=float(ret.std(ddof=1)),
                           reward_range_mean=float(((ret+50)/50*100 if family=="fetch" else (ret+20)/2020*100).mean())))
summary = dict(at=D["at"], stamp=STAMP, scientific_counts=dict(collections.Counter(r["job"]["status"] for r in D["rows"])),
               warning="iBOLT pilot seeds 5,6 versus JAX SAC seeds 0-4; exploratory, not paired-seed confirmation. Missing seeds never filled with zeros.",
               selected_temperatures=D["selections"], shared_configuration_checks=checks, fixed_step_table=tables)
(P / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
