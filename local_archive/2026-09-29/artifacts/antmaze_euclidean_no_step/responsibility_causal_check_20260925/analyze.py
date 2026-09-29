#!/usr/bin/env python3
"""Read-only decomposition of one saved AntMaze v3/v4 training run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ndtr, logsumexp

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/antmaze_euclidean_no_step"
OUT = BASE / "responsibility_causal_check_20260925"
PROBE = BASE / "mode_q_probe/remote"
LEARNER = BASE / "saved_evidence_20260925/v3-learner.csv"
ROLLOUTS = BASE / "saved_evidence_20260925/saved-rollout-metrics.json"
OUT.mkdir(parents=True, exist_ok=True)

def component_responsibility(npz):
    z = np.load(npz)
    a = z["candidates"].astype(np.float64)
    mu = z["mu"][0].astype(np.float64)
    log_std = z["logstd"][0].astype(np.float64)
    std = np.exp(log_std)
    log_norm = np.log(ndtr((1 - mu) / std) - ndtr((-1 - mu) / std))
    delta = (a[None, :, :] - mu[:, None, :]) / std[:, None, :]
    ell = (-0.5 * delta**2 - log_std[:, None, :]
           - 0.5 * np.log(2 * np.pi) - log_norm[:, None, :]).sum(-1)
    resp = np.exp(ell - logsumexp(ell, axis=0, keepdims=True))
    outcomes = z["outcomes"]
    route_p = {str(r): (outcomes == r).mean(0).astype(float)
               for r in dict.fromkeys(outcomes.ravel().tolist())}
    weights = {"proposal": np.full(len(a), 1 / len(a)),
               "Q only": z["q_only_weights"].astype(float),
               "Q + density": z["weights"].astype(float)}
    result = {"candidate_count": int(len(a)), "component_count": int(len(mu)),
              "route_probability_from_four_continuations": {}, "weights": {},
              "responsibility_effective_components": {}}
    for route, p in route_p.items():
        result["route_probability_from_four_continuations"][route] = float(p.mean())
    for name, w in weights.items():
        result["weights"][name] = {}
        result["responsibility_effective_components"][name] = {}
        for route, p in route_p.items():
            wp = w * p
            mass = wp.sum()
            result["weights"][name][route] = float(mass)
            if mass <= 1e-12:
                continue
            usage = (resp * wp[None, :]).sum(1) / mass
            ess = 1.0 / np.square(usage).sum()
            result["responsibility_effective_components"][name][route] = {
                "effective_components": float(ess),
                "max_component_fraction": float(usage.max()),
                "top8_component_fractions": np.sort(usage)[-8:][::-1].tolist(),
            }
    usage = {name: (resp * w[None, :]).sum(1) for name, w in weights.items()}
    result["responsibility_effective_components"]["all_candidates"] = {
        name: float(1 / np.square(x / x.sum()).sum()) for name, x in usage.items()
    }
    return result

def main():
    route = {e: component_responsibility(PROBE / e / "teacher_probe.npz")
             for e in ("v3", "v4")}
    df = pd.read_csv(LEARNER)
    roll = json.loads(ROLLOUTS.read_text())
    run = next(x for x in roll["runs"] if x["task"] == "v3")
    windows = [(0,250112),(250112,500224),(500224,750080)]
    trace=[]
    for lo,hi in windows:
        x=df[(df["time/total_timesteps"]>lo)&(df["time/total_timesteps"]<=hi)]
        trace.append({"start_step_exclusive":lo,"end_step_inclusive":hi,
            "rows":int(len(x)),
            "source_ess":float(x["train/source_ess_absolute"].median()),
            "component_ess_fraction":float(x["train/gmm_component_ess_fraction"].median()),
            "component_ess":float(64*x["train/gmm_component_ess_fraction"].median()),
            "underused_fraction":float(x["train/gmm_underused_fraction"].median()),
            "max_source_weight":float(x["train/max_source_weight"].median()),
            "source_q_std":float(x["train/source_q_std"].median())})
    result={"training_source":"f953d28456d3800860dddb9b9cb91b6bd520ae00",
      "training_run":"antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2",
      "scope":"read-only post-hoc; per-state teacher/responsibility probe at 250112; v3 learner metrics through 750080; fixed-start policy evaluations at 250112/500224/750080",
      "teacher_probe":route,
      "v3_training_component_metrics":trace,
      "v3_policy_fixed_evaluation_routes":[
        {"step":250112,"episodes":40,"left":12,"right":23,"uncommitted":5},
        {"step":500224,"episodes":40,"left":0,"right":39,"uncommitted":1},
        {"step":750080,"episodes":40,"left":0,"right":39,"uncommitted":1}],
      "limits":["one seed","one candidate cloud and four continuations per task for teacher probe","global component metrics average over replay minibatches and are not branch-conditioned","checkpoint probe is at 250112; no same-origin component-responsibility probe at later checkpoints"]}
    (OUT/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")

    fig,axs=plt.subplots(1,3,figsize=(16,5.2),constrained_layout=True)
    colors={"proposal":"#78909c","Q only":"#ef8a62","Q + density":"#2878b5"}
    for ax,e,routes in [(axs[0],"v3",["left","right"]),(axs[1],"v4",["lower","upper"])]:
        data=route[e]; names=list(colors); x=np.arange(len(routes)); width=.24
        for j,name in enumerate(names):
            vals=[data["weights"][name].get(r,0)*100 for r in routes]
            ax.bar(x+(j-1)*width,vals,width,label=name,color=colors[name])
        ax.set_xticks(x,routes); ax.set_ylim(0,100); ax.set_ylabel("Teacher mass (%)")
        ax.set_title(f"{e}: 250k fixed-start proposal → teacher")
        ax.grid(axis="y",alpha=.22)
    axs[0].legend(frameon=False,fontsize=8)
    ax=axs[2]
    steps=["250k","500k","750k"]
    route_left=np.array([12,0,0])/40*100
    route_right=np.array([23,39,39])/40*100
    route_other=np.array([5,1,1])/40*100
    ax.bar(steps,route_right,color="#2878b5",label="right eval route")
    ax.bar(steps,route_left,bottom=route_right,color="#ef8a62",label="left eval route")
    ax.bar(steps,route_other,bottom=route_right+route_left,color="#bdbdbd",label="uncommitted")
    ax.set_ylim(0,100); ax.set_ylabel("40-episode route fraction (%)")
    ax.set_title("v3 rollout concentration vs global NLL use")
    ax.grid(axis="y",alpha=.22)
    comp=[t["component_ess"] for t in trace]
    ax2=ax.twinx(); ax2.plot(steps,comp,color="#6a3d9a",marker="o",lw=2,label="effective GMM components")
    ax2.set_ylim(0,64); ax2.set_ylabel("Effective components / 64",color="#6a3d9a")
    ax2.tick_params(axis="y",colors="#6a3d9a")
    lines,labels=ax.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
    ax.legend(lines+lines2,labels+labels2,fontsize=7,loc="lower left",frameon=True)
    fig.suptitle("Observed Q weighting, GMM responsibilities, and route collapse (single seed)",fontsize=13)
    fig.savefig(OUT/"responsibility_causal_check.png",dpi=180)
    print(json.dumps({"out":str(OUT),"trace":trace,
      "v3_masses":route["v3"]["weights"],
      "v3_resp":route["v3"]["responsibility_effective_components"],
      "v4_masses":route["v4"]["weights"],
      "v4_resp":route["v4"]["responsibility_effective_components"]},indent=2))

if __name__=="__main__": main()
