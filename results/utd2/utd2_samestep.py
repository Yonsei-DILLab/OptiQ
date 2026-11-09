"""UTD=2 vs UTD=1 at the same env step per task (common step = slowest UTD2 run of that task, excl. no-DACER)."""
import glob, os, re, numpy as np
from collections import defaultdict
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utd2_status.py")).read().split("G = defaultdict")[0])  # reuse ev(), at(), H
G = defaultdict(list)
for d in glob.glob(f"{H}/outputs_utd2/*"):
    G[re.sub(r"-s\d+_.*$", "", os.path.basename(d))].append(d)
U1 = {"humanoid": [("CrossQ", "sbx", [f"{H}/results/live_refs/humanoid-crossq-q1024-s*"]),
                   ("iBOLT T=0.5", "ib", [f"{H}/outputs_tsweep/humanoid-ibolt-crossqAC-T0.5-s*", f"{H}/outputs_pc/humanoid-ibolt-crossqAC-T0.5-s*"])],
      "walker2d": [("CrossQ", "sbx", [f"{H}/outputs/walker2d-crossq-q1024-s*", "/tmp/hw/4090/walker2d-crossq-q1024-s*"]),
                   ("iBOLT T=0.1", "ib", [f"{H}/outputs_tsweep/walker2d-ibolt-crossqAC-T0.1-s*"])],
      "ant": [("CrossQ", "sbx", [f"{H}/outputs/ant-crossq-q1024-s*"])]}
def row(lab, dirs, kind, step):
    ms = ["deterministic", None, "stochastic"] if kind == "sbx" else ["zero_z", "stochastic_z", "stochastic_z_sigma"]
    out, n = [], 0
    for m in ms:
        if m is None: out.append("      —"); continue
        v = [at(r, step) for r in (ev(d, m) for d in dirs) if r is not None]
        v = [x for x in v if not np.isnan(x)]; n = max(n, len(v))
        out.append(f"{np.mean(v):7.0f}" if v else "      ·")
    print(f"  {lab:30s} n={n}  {' '.join(out)}")
for task in ["humanoid", "walker2d", "ant"]:
    gs = [g for g in G if g.startswith(task + "-")]
    def cur(g): return min(int(r[0][-1]) for r in (ev(d, "deterministic" if "-crossq-q" in g else "zero_z") for d in G[g]) if r is not None)
    step = min(cur(g) for g in gs if "nodacer" not in g) // 5000 * 5000
    print(f"== {task} @ {step//1000}K steps (25K window)        z0/det stoch z  z+σ/smp")
    for g in sorted(gs, key=lambda s: ("crossq-q" not in s, "nodacer" in s, s)):
        kind = "sbx" if "-crossq-q" in g else "ib"
        lab = "UTD2 CrossQ" if kind == "sbx" else "UTD2 iBOLT " + g.split("utd2-")[1].replace("nodacer-", "noDACER ")
        if "nodacer" in g and cur(g) < step: lab += f" (only {cur(g)//1000}K)"
        row(lab, G[g], kind, step)
    for lab, kind, pats in U1[task]:
        row("UTD1 " + lab, [d for p in pats for d in glob.glob(p)], kind, step)
