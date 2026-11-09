"""Live learning curves (Humanoid, HalfCheetah): CrossQ-actor runs vs references, re-read on every refresh.

Run on the 5090:  ~/OptiQ/.venv/bin/python live_curves.py [port=8765]   (binds 127.0.0.1)
View locally:     ssh -N -L 8765:localhost:8765 vast-5090   then open http://localhost:8765
Live runs:  ~/OptiQ/outputs_tsweep/<task>-ibolt-crossqAC-T0.5-s*   (eval npz written every 5K steps)
References (copied once into ~/OptiQ/results/live_refs by results/tsweep/live.sh):
  SBX CrossQ (5 seeds), iBOLT + CrossQ critic T=0.5, paper-critic iBOLT (Humanoid T=0.1, HalfCheetah T=0.25).
"""
import glob, io, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~/OptiQ")
LIVE = os.path.join(HOME, "outputs_tsweep")
LIVE_PC = os.path.join(HOME, "outputs_pc")  # eval dirs copied from the local RTX 5070 Ti (WSL) every minute
REF = os.path.join(HOME, "results/live_refs")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
TASKS = [("humanoid", "Humanoid-v4", "humanoid-ibolt-T0.1-s*", "paper critic T=0.1"),
         ("halfcheetah", "HalfCheetah-v4", "halfcheetah-ibolt-T0.25-s*", "paper critic T=0.25")]
MODES = [("zero_z", "deterministic", "z = 0"), ("stochastic_z", None, "stochastic z (center)"),
         ("stochastic_z_sigma", "stochastic", "z + σ (full policy)")]
INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"
LIVE_COLORS = ["#111111", "#7a3fbf", "#c2185b", "#0b6e99"]


def load(pattern, mode):
    runs = []
    for d in sorted(glob.glob(pattern)):
        f = glob.glob(os.path.join(d, "**", f"evaluations_{mode}.npz"), recursive=True)
        if not f:
            continue
        try:
            z = np.load(f[0])
            t, y = np.asarray(z["timesteps"]), np.asarray(z["results"]).mean(1)
        except Exception:  # file being replaced right now
            continue
        if len(t):
            runs.append((t, y, os.path.basename(d).split("_")[0]))
    return runs


def mean_curve(runs, k=5):
    L = max(len(t) for t, _, _ in runs)
    t = max(runs, key=lambda r: len(r[0]))[0]
    Y = np.full((len(runs), L), np.nan)
    for i, (_, y, _) in enumerate(runs):
        Y[i, :len(y)] = y
    keep = (~np.isnan(Y)).sum(0) >= min(2, len(runs))
    mu = np.nanmean(Y[:, keep], 0)
    return t[keep], smooth(mu, k)


def smooth(y, k=5):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def render():
    fig, axes = plt.subplots(2, 3, figsize=(19, 9), facecolor=SURF)
    lines = []
    for row, (task, title, paper_pat, paper_lab) in enumerate(TASKS):
        for col, (mode, sbx_mode, mlab) in enumerate(MODES):
            ax = axes[row, col]
            ax.set_facecolor(SURF)
            for runs, color, ls, lab in [
                (load(os.path.join(REF, f"{task}-crossq-q*"), sbx_mode) if sbx_mode else [], "#eb6834", "--", "CrossQ (SBX)"),
                (load(os.path.join(REF, f"{task}-ibolt-crossq-T0.5-s*"), mode), "#1a9e3f", "-", "CrossQ critic, T=0.5"),
                (load(os.path.join(REF, paper_pat), mode), "#8a8985", ":", paper_lab)]:
                if runs:
                    t, mu = mean_curve(runs)
                    ax.plot(t / 1e6, mu, color=color, linestyle=ls, linewidth=1.8, label=f"{lab} ({len(runs)})")
            live = [r + ("",) for r in load(os.path.join(LIVE, f"{task}-ibolt-crossqAC-T0.5-s*"), mode)]
            live += [r + (" (PC)",) for r in load(os.path.join(LIVE_PC, f"{task}-ibolt-crossqAC-T0.5-s*"), mode)]
            for i, (t, y, name, where) in enumerate(live):
                ax.plot(t / 1e6, smooth(y), color=LIVE_COLORS[i % len(LIVE_COLORS)], linewidth=2.2,
                        label=f"+ CrossQ actor training, s{name.rsplit('-s', 1)[1]}{where}")
                if col == 0:
                    lines.append(f"{task} s{name.rsplit('-s', 1)[1]}{where}: {int(t[-1]) // 1000}K steps, "
                                 f"last eval z=0 {y[-1]:.0f}, last-25K {y[-5:].mean():.0f}")
            ax.set_title(f"{title} — {mlab}", fontsize=10, loc="left")
            ax.set_xlim(0, 1)
            ax.grid(True, color=GRID, linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if row == 1:
                ax.set_xlabel("Environment steps (M)")
            if col == 0:
                ax.set_ylabel("Evaluation return (25K moving avg)")
            ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"iBOLT + CrossQ critic + CrossQ actor training (T=0.5) — live, {time.strftime('%H:%M:%S')}",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=SURF)
    plt.close(fig)
    return buf.getvalue(), lines


CACHE = {"t": 0.0, "png": b"", "lines": []}
LOCK = threading.Lock()


def cached():
    with LOCK:
        if time.time() - CACHE["t"] > 20:
            CACHE["png"], CACHE["lines"] = render()
            CACHE["t"] = time.time()
        return CACHE["png"], CACHE["lines"]


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/plot.png"):
            body, ctype = cached()[0], "image/png"
        elif self.path in ("/", "/index.html"):
            lines = "".join(f"<li>{l}</li>" for l in cached()[1])
            body = (f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=60>"
                    f"<title>Live curves</title><body style='margin:12px;font:13px system-ui;background:{SURF}'>"
                    f"<div style='color:{INK2}'>auto-refresh every 60 s · evals every 5K steps</div>"
                    f"<ul>{lines}</ul><img src='/plot.png?t={time.time():.0f}' style='max-width:100%'>"
                    ).encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
