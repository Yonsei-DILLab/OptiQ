#!/usr/bin/env python3
"""Add one sampled SNIS-NLL trajectory to frozen exact-KL baselines."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
from reproduce import Objective, mixture, save_csv, write_json, plt, matplotlib

HERE = Path(__file__).resolve().parent


def sample_q(theta, n, rng, weights):
    component = rng.choice(2, size=n, p=weights)
    return theta[component] + theta[2 + component] * rng.standard_normal(n)


def normalize_log_weights(log_weights):
    w = np.exp(log_weights - np.max(log_weights))
    return w / w.sum()


def snis_weights(theta, samples, config):
    weights = np.asarray(config["weights"])
    logp, _ = mixture(samples, np.array([0., config["a"]]),
                      np.array(config["target_sigma"]), weights)
    logq, _ = mixture(samples, theta[:2], theta[2:], weights)
    return normalize_log_weights(logp - logq)


def frozen_nll(theta, samples, importance_weights, mixture_weights):
    """Partial derivative in theta ONLY; samples and weights are fixed arrays."""
    mu, sigma = theta[:2], theta[2:]
    if not np.isfinite(theta).all() or np.any(sigma <= 0):
        raise FloatingPointError(f"Invalid parameters: {theta}")
    logq, resp = mixture(samples, mu, sigma, mixture_weights)
    delta = samples[:, None] - mu
    weighted_resp = importance_weights[:, None] * resp
    grad = -np.concatenate((
        np.sum(weighted_resp * delta / sigma**2, axis=0),
        np.sum(weighted_resp * (delta**2 / sigma**3 - 1/sigma), axis=0)))
    loss = float(-np.sum(importance_weights * logq))
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise FloatingPointError("Nonfinite sampled NLL or gradient")
    return loss, grad


def validate(config, settings):
    rng = np.random.default_rng(271828)  # Never shares the training RNG.
    rows = []
    n = settings["samples_per_update"]
    mw = np.asarray(config["weights"])
    for theta in (np.array([0., 2.5, 1., 1.]), np.array([-.07, 5.42, .85, 4.81])):
        x = sample_q(theta, n, rng, mw)
        iw = snis_weights(theta, x, config)
        _, gradient = frozen_nll(theta, x, iw, mw)
        fd = np.zeros(4)
        for j in range(4):
            h = 1e-5 * max(1., abs(theta[j]))
            offset = np.zeros(4)
            offset[j] = h
            fd[j] = (frozen_nll(theta+offset, x, iw, mw)[0]
                     - frozen_nll(theta-offset, x, iw, mw)[0]) / (2*h)
        np.testing.assert_allclose(gradient, fd, rtol=2e-6, atol=2e-7)
        assert abs(iw.sum()-1) < 1e-12
        ess = float(1 / np.sum(iw**2))
        assert 1-1e-12 <= ess <= n+1e-10
        # An additive target log-normalizer must not change normalized weights.
        np.testing.assert_allclose(normalize_log_weights(np.log(iw)+123), iw,
                                   rtol=1e-12, atol=1e-14)
        rows.append({"theta": theta.tolist(), "ess": ess,
                     "max_abs_gradient_error": float(np.max(np.abs(gradient-fd)))})
    target_theta = np.array([0., config["a"], *config["target_sigma"]])
    x = sample_q(target_theta, n, rng, mw)
    np.testing.assert_allclose(snis_weights(target_theta, x, config), np.full(n, 1/n),
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(
        sample_q(target_theta, n, np.random.default_rng(123), mw),
        sample_q(target_theta, n, np.random.default_rng(123), mw))
    return {"frozen_nll_finite_difference_checks": rows,
            "weights_normalized": True, "ess_in_bounds": True,
            "target_log_normalizer_invariant": True, "uniform_weights_when_p_equals_q": True,
            "seeded_sampling_reproducible": True}


def train(config, settings, obj, out):
    rng = np.random.default_rng(settings["seed"])
    theta = np.array(config["initial_mu"] + config["initial_sigma"], dtype=np.float64)
    mw = np.asarray(config["weights"])
    trace, snapshots = [], {}
    n = settings["samples_per_update"]
    for step in range(config["steps"]+1):
        if step in config["snapshots"]:
            snapshots[str(step)] = {"theta": theta.tolist(), **obj.metrics(theta)}
            state = snapshots[str(step)]
            print(f"SNIS iter={step:4d} mu={theta[:2]} sigma={theta[2:]} "
                  f"exact_forward_KL={state['forward_kl']:.8g} "
                  f"right_mass={state['right_mode_mass']:.8g}", flush=True)
        if step == config["steps"]:
            break
        # Samples, proposal log density and weights are computed at pre-update theta.
        # frozen_nll differentiates only the evaluated student mixture parameters.
        x = sample_q(theta, n, rng, mw)
        iw = snis_weights(theta, x, config)
        loss, grad = frozen_nll(theta, x, iw, mw)
        trace.append({"iteration": step, "sampled_snis_nll": loss,
                      "mu1": theta[0], "mu2": theta[1], "sigma1": theta[2], "sigma2": theta[3],
                      "gradient_norm": float(np.linalg.norm(grad)),
                      "ess": float(1/np.sum(iw**2)), "max_weight": float(iw.max()),
                      "sample_right_count": int(np.sum(x > config["a"]/2)),
                      "sample_min": float(x.min()), "sample_max": float(x.max())})
        theta -= config["learning_rate"] * grad
        if not np.isfinite(theta).all() or np.any(theta[2:] <= 0):
            save_csv(out / "failed_trajectory.csv", trace)
            raise FloatingPointError(f"Invalid update {step+1}: {theta}")
    save_csv(out / "snis_trajectory.csv", trace)
    write_json(out / "final_rng_state.json", rng.bit_generator.state)
    return snapshots, trace


def draw(config, settings, snapshots, trace, out):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "mathtext.fontset": "dejavusans"})
    x = np.linspace(config["plot_min"], config["plot_max"], 1801)
    mw = np.asarray(config["weights"])
    logp, _ = mixture(x, np.array([0., config["a"]]), np.array(config["target_sigma"]), mw)
    fig, axes = plt.subplots(len(config["snapshots"]), 3, figsize=(16.2, 13.4),
                             sharex=True, sharey=True)
    titles = ["Forward KL", "Reverse KL", "Sampling + SNIS NLL\n"
              f"256 samples/update, seed {settings['seed']}"]
    titles[2] = titles[2].replace("256", str(settings["samples_per_update"]))
    for col, direction in enumerate(("forward", "reverse", "snis")):
        axes[0, col].set_title(titles[col])
        for row, step in enumerate(config["snapshots"]):
            ax = axes[row, col]
            theta = np.asarray(snapshots[direction][str(step)]["theta"])
            logq, resp = mixture(x, theta[:2], theta[2:], mw)
            q = np.exp(logq)
            ax.plot(x, np.exp(logp), color="tab:blue", lw=1.6, label="p")
            ax.plot(x, q, "--", color="tab:orange", lw=1.8, label="q")
            ax.plot(x, q[:, None]*resp, color="tab:orange", lw=.7, alpha=.45)
            ax.text(.02, .94, f"iter {step}:  $\\mu$=({theta[0]:.2f}, {theta[1]:.2f}),  "
                    f"$\\sigma$=({theta[2]:.2f}, {theta[3]:.2f})",
                    transform=ax.transAxes, va="top", fontsize=9)
            ax.set(ylim=(0,.45), yticks=[0,.2,.4], xlim=(-4.9,14.9))
            ax.set_xticks(np.arange(-2.5,13,2.5))
            ax.grid(alpha=.25)
            if row == len(config["snapshots"])-1:
                ax.set_xlabel("x")
    axes[0,0].legend(loc="upper right", fontsize=9)
    fig.suptitle(r"$a=10$, init $(\mu_1,\mu_2,\sigma_1,\sigma_2)=(0,a/4,1,1)$, "
                 "gradient descent lr=0.02", y=.995, fontsize=14)
    fig.tight_layout(rect=[0,0,1,.975])
    fig.savefig(out/"reproduction_snis.png", dpi=180)
    fig.savefig(out/"reproduction_snis.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13.5,3.5))
    steps = [row["iteration"] for row in trace]
    for ax, key, label in zip(axes, ["sampled_snis_nll", "ess", "sample_right_count"],
                             ["Sampled SNIS NLL (fresh batch)", "SNIS effective sample size", "Proposal samples with x > 5"]):
        ax.plot(steps, [r[key] for r in trace], lw=.6)
        ax.set(xlabel="Pre-update iteration", title=label)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"snis_diagnostics.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-inputs", type=Path, required=True)
    parser.add_argument("--source-commit", default="unversioned")
    parser.add_argument("--run-id", default="standalone")
    args = parser.parse_args()
    config = json.loads((HERE/"config.json").read_text())
    settings = json.loads((HERE/"snis_config.json").read_text())
    bp = args.baseline_inputs/"baseline_provenance.json"
    bs = args.baseline_inputs/"baseline_snapshots.json"
    baseline_provenance = json.loads(bp.read_text())
    assert baseline_provenance["source_commit"] == settings["baseline_source_commit"]
    assert baseline_provenance["config"] == config
    assert config["parameterization"] == "sigma"
    assert settings["proposal"] == "current_q" and settings["sampling"] == "iid_mixture"
    assert settings["stop_gradient_samples"] and settings["stop_gradient_weights"]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = {
        "source_commit": args.source_commit, "run_id": args.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "config": config, "snis_config": settings,
        "baseline_source_commit": baseline_provenance["source_commit"],
        "baseline_input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (bp,bs)},
        "source_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(HERE.iterdir()) if p.is_file()}}
    write_json(out/"PROVENANCE.json", provenance)
    validation = validate(config, settings)
    write_json(out/"validation.json", validation)
    obj = Objective(config)
    snis, trace = train(config, settings, obj, out)
    baseline = json.loads(bs.read_text())
    snapshots = {**baseline, "snis": snis}
    write_json(out/"snapshots.json", snapshots)
    assert {k:snapshots[k] for k in ("forward","reverse")} == baseline
    validation["baseline_columns_unchanged"] = True
    validation["all_training_values_finite_and_scales_positive"] = True
    write_json(out/"validation.json", validation)
    draw(config, settings, snapshots, trace, out)
    summary = {direction: rows[str(config["steps"])] for direction,rows in snapshots.items()}
    summary["sampling"] = {**settings, "total_training_samples": len(trace)*settings["samples_per_update"],
        "mean_ess": float(np.mean([r["ess"] for r in trace])),
        "min_ess": min(r["ess"] for r in trace),
        "batches_with_right_samples": sum(r["sample_right_count"]>0 for r in trace)}
    write_json(out/"summary.json", summary)
    provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["elapsed_seconds"] = time.perf_counter()-started
    write_json(out/"PROVENANCE.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
