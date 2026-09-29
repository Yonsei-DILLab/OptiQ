#!/usr/bin/env python3
"""Counterfactual reweighting of a saved single teacher cloud; no learning."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "artifacts/antmaze_euclidean_no_step/mode_q_probe/remote"
OUT = ROOT / "artifacts/antmaze_euclidean_no_step/responsibility_causal_check_20260925"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    grid = np.geomspace(.05, 1000, 3000)
    rows = {}
    for env, minority in (("v3", "left"), ("v4", "lower")):
        z=np.load(PROBE/env/"teacher_probe.npz")
        q=z["q"].mean(0).astype(float)
        logq=z["logq"].astype(float)
        outcomes=z["outcomes"]
        routep={str(r):(outcomes==r).mean(0).astype(float)
                for r in dict.fromkeys(outcomes.ravel().tolist())}
        base=float(routep[minority].mean())
        def mass(t,beta=1.0):
            w=softmax(q/t-beta*logq)
            return float(w@routep[minority])
        roots=[]
        prev=grid[0]; fprev=mass(prev)-base
        for t in grid[1:]:
            f=mass(t)-base
            if f*fprev<0:
                roots.append(float(brentq(lambda u:mass(u)-base,prev,t)))
            prev,fprev=t,f
        points=sorted(set([.25,.5,1,1.25,1.5,2,3,5,10,100,1000]+[round(r,6) for r in roots]))
        vals=[]
        for t in points:
            w=softmax(q/t-logq)
            vals.append({"temperature":t,"minority_teacher_mass":float(w@routep[minority]),
                         "proposal_minority_mass":base,"source_ess":float(1/(w@w)),
                         "q_only_minority_mass":float(softmax(q/t)@routep[minority])})
        vals_grid=np.array([mass(t) for t in grid])
        hi=grid>=1
        rows[env]={"minority_route":minority,"density_beta":1.0,
                   "candidate_count":int(len(q)),"proposal_route_mass":base,
                   "temperatures_matching_proposal_mass":roots,
                   "max_minority_mass_for_T_ge_1":float(vals_grid[hi].max()),
                   "max_mass_temperature_for_T_ge_1":float(grid[hi][np.argmax(vals_grid[hi])]),
                   "temperature_points":vals}
    (OUT/"temperature_counterfactual.json").write_text(json.dumps({
        "source_commit":"f953d28456d3800860dddb9b9cb91b6bd520ae00",
        "method":"Reweight saved 64-candidate teacher clouds with softmax(Q/T - log q), beta=1; no environment steps or optimizer updates.",
        "limits":["one candidate cloud per maze at 250112 steps","4 paired continuations define the empirical route probability","counterfactual changes weights only; it does not rerun actor learning or create unsupported actions"],
        "tasks":rows},indent=2)+"\n")
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,(env,minority) in zip(axs,(("v3","left"),("v4","lower"))):
        z=np.load(PROBE/env/"teacher_probe.npz")
        q=z["q"].mean(0).astype(float);logq=z["logq"].astype(float)
        p=(z["outcomes"]==minority).mean(0).astype(float)
        base=float(p.mean()); mass=[]; ess=[]; qonly=[]
        for t in grid:
            w=softmax(q/t-logq);mass.append(w@p);ess.append(1/(w@w))
            qonly.append(softmax(q/t)@p)
        ax.plot(grid,mass,label="Q + density (β=1)",lw=2,color="#2878b5")
        ax.plot(grid,qonly,label="Q only (β=0)",lw=1.7,color="#ef8a62")
        ax.axhline(base,color="#333333",ls="--",lw=1.2,label="current proposal fraction")
        ax.scatter([1],[float(z["weights"]@(p))],color="#2878b5",zorder=5)
        ax.set_xscale("log");ax.set_ylim(0,1);ax.set_xlabel("Teacher temperature T")
        ax.set_ylabel(f"{minority} route mass after weighting")
        ax.set_title(f"{env}: fixed 250k candidate cloud")
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Offline temperature sweep; proposal and candidate actions are held fixed")
    fig.savefig(OUT/"temperature_counterfactual.png",dpi=180)
    print((OUT/"temperature_counterfactual.json").read_text())

if __name__=="__main__":main()
