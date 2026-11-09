"""Policy width from logs: SAC (SBX CrossQ) entropy recovered from ent_coef_loss = log(alpha) * (H - H_target),
H_target = -|A|  ->  H = ent_coef_loss / log(alpha) - |A|  (action-space entropy of tanh-Gaussian samples).
iBOLT: exploration/entropy_proxy (GMM fit to 200 behaviour actions per state, action space).
sigma_eq = Gaussian-equivalent per-dim std with the same per-dim entropy: exp(H/d - 0.5*ln(2*pi*e))."""
import glob, csv, os, sys, numpy as np
H = os.path.expanduser("~/OptiQ")
D = dict(walker2d=6, humanoid=17, halfcheetah=6)
IB = dict(walker2d="walker2d-ibolt-crossqAC-T0.1", humanoid="humanoid-ibolt-crossqAC-T0.5", halfcheetah="halfcheetah-ibolt-crossqAC-T0.5")
STEPS = [25_000, 50_000, 100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000]
c = 0.5 * np.log(2 * np.pi * np.e)
def series(f, xk, fn):
    r = list(csv.DictReader(open(f))); xs, ys = [], []
    for a in r:
        try: x0, y0 = float(a[xk]), fn(a); xs.append(x0); ys.append(y0)
        except (ValueError, KeyError, TypeError, ZeroDivisionError): pass
    return np.array(xs), np.array(ys)
def at(rows, s, w=25_000):
    v = [y[(x > s - w) & (x <= s) & np.isfinite(y)].mean() for x, y in rows if ((x > s - w) & (x <= s) & np.isfinite(y)).any()]
    return (np.mean(v), len(v)) if v else (np.nan, 0)
for task, d in D.items():
    sb = [f for f in glob.glob(f"{H}/outputs*/{task}-crossq-q*-s*/logs/progress.csv") if "utd2" not in f]
    sac = [series(f, "time/total_timesteps", lambda a: float(a["train/ent_coef_loss"]) / np.log(float(a["train/ent_coef"])) - d) for f in sb]
    ib = [series(f, "env_steps", lambda a: float(a["exploration/entropy_proxy"])) for f in glob.glob(f"{H}/outputs*/{IB[task]}-s*/logs/progress.csv")]
    for name, rows in (("SAC", sac), ("iBOLT", ib)):
        rows = [r for r in rows if len(r[0])]
        if not rows: continue
        cells = []
        for s in STEPS:
            h, n = at(rows, s)
            cells.append(f"{h/d:+.2f}/{np.exp(h/d - c):.3f}" if n else "      —     ")
        print(f"{task:11s} {name:5s} n={len(rows)}  " + "  ".join(cells))
print("cells: entropy per dim (nats) / sigma_eq;  steps:", " ".join(f"{s//1000}K" for s in STEPS))
print(f"reference: uniform on [-1,1] per dim = {np.log(2):+.2f}/{np.exp(np.log(2)-c):.3f};  SAC target = -1.00/{np.exp(-1-c):.3f}")
