#!/usr/bin/env python3
"""Deterministic gradient descent for two-component forward and reverse KL."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_2PI = math.log(2 * math.pi)
HERE = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def mixture(x, mu, sigma, weights):
    """Return log density and normalized component responsibilities."""
    delta = x[:, None] - mu
    log_components = (np.log(weights) - np.log(sigma)
                      - 0.5 * LOG_2PI - 0.5 * (delta / sigma) ** 2)
    log_density = np.logaddexp(log_components[:, 0], log_components[:, 1])
    return log_density, np.exp(log_components - log_density[:, None])


class Objective:
    def __init__(self, config, refinement=1):
        self.config = config
        self.weights = np.asarray(config["weights"], dtype=np.float64)
        lo, hi = config["integration_min"], config["integration_max"]
        intervals = round((hi - lo) / config["integration_dx"]) * refinement
        self.x = np.linspace(lo, hi, intervals + 1, dtype=np.float64)
        self.w = np.full(intervals + 1, (hi - lo) / intervals)
        self.w[[0, -1]] *= 0.5
        self.logp, _ = mixture(self.x, np.array([0.0, config["a"]]),
                               np.array(config["target_sigma"]), self.weights)
        self.p = np.exp(self.logp)

    def evaluate(self, theta, direction):
        mu, sigma = theta[:2], theta[2:]
        if np.any(sigma <= 0) or not np.isfinite(theta).all():
            raise FloatingPointError(f"Invalid parameters: {theta}")
        logq, resp = mixture(self.x, mu, sigma, self.weights)
        delta = self.x[:, None] - mu
        score_mu = resp * delta / sigma**2
        score_sigma = resp * (delta**2 / sigma**3 - 1 / sigma)
        if direction == "forward":
            loss = np.sum(self.w * self.p * (self.logp - logq))
            factor = -self.w * self.p
        elif direction == "reverse":
            q = np.exp(logq)
            loss = np.sum(self.w * q * (logq - self.logp))
            factor = self.w * q * (logq - self.logp + 1)
        else:
            raise ValueError(direction)
        grad = np.concatenate((np.sum(factor[:, None] * score_mu, axis=0),
                               np.sum(factor[:, None] * score_sigma, axis=0)))
        if not np.isfinite(loss) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite loss or gradient")
        return float(loss), grad

    def metrics(self, theta):
        logq, _ = mixture(self.x, theta[:2], theta[2:], self.weights)
        q = np.exp(logq)
        right_mass = sum(float(w) * 0.5 * math.erfc(
            (self.config["a"] / 2 - float(mu)) / (math.sqrt(2) * float(sigma)))
            for w, mu, sigma in zip(self.weights, theta[:2], theta[2:]))
        return {
            "forward_kl": self.evaluate(theta, "forward")[0],
            "reverse_kl": self.evaluate(theta, "reverse")[0],
            "tv": float(0.5 * np.sum(self.w * np.abs(self.p - q))),
            "right_mode_mass": right_mass,
            "integrated_q_mass": float(np.sum(self.w * q)),
        }


def validate_gradients(obj):
    rows = []
    states = [np.array([0., 2.5, 1., 1.]), np.array([-.07, 5.42, .85, 4.81])]
    for state in states:
        for direction in ("forward", "reverse"):
            _, grad = obj.evaluate(state, direction)
            numerical = np.empty(4)
            for j in range(4):
                h = 1e-5 * max(1., abs(state[j]))
                d = np.zeros(4)
                d[j] = h
                numerical[j] = (obj.evaluate(state + d, direction)[0]
                                - obj.evaluate(state - d, direction)[0]) / (2*h)
            err = float(np.max(np.abs(grad - numerical)))
            np.testing.assert_allclose(grad, numerical, rtol=2e-6, atol=2e-7)
            mass = obj.metrics(state)["integrated_q_mass"]
            assert abs(mass - 1) < 1e-9
            rows.append({"theta": state.tolist(), "direction": direction,
                         "max_abs_gradient_error": err, "q_mass": mass})
    assert abs(np.sum(obj.w * obj.p) - 1) < 1e-12
    return rows


def train(obj, config, direction):
    if config["parameterization"] != "sigma":
        raise ValueError("This committed protocol optimizes sigma directly.")
    theta = np.array(config["initial_mu"] + config["initial_sigma"], dtype=np.float64)
    snapshots, trace = {}, []
    for step in range(config["steps"] + 1):
        loss, gradient = obj.evaluate(theta, direction)
        trace.append({"direction": direction, "iteration": step,
                      "loss": loss, "mu1": theta[0], "mu2": theta[1],
                      "sigma1": theta[2], "sigma2": theta[3],
                      "gradient_norm": float(np.linalg.norm(gradient))})
        if step in config["snapshots"]:
            snapshots[step] = {"theta": theta.copy(), **obj.metrics(theta)}
            print(f"{direction:7s} iter={step:4d} mu={theta[:2]} sigma={theta[2:]} "
                  f"KL={loss:.10g}", flush=True)
        if step < config["steps"]:
            theta = theta - config["learning_rate"] * gradient
    return snapshots, trace


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(config, snapshots, traces, out):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "mathtext.fontset": "dejavusans"})
    x = np.linspace(config["plot_min"], config["plot_max"], 1801)
    weights = np.asarray(config["weights"])
    lp, _ = mixture(x, np.array([0., config["a"]]),
                    np.array(config["target_sigma"]), weights)
    fig, axes = plt.subplots(len(config["snapshots"]), 2, figsize=(10.8, 13.4),
                             sharex=True, sharey=True)
    for col, direction in enumerate(("forward", "reverse")):
        axes[0, col].set_title(f"{direction.title()} KL")
        for row, step in enumerate(config["snapshots"]):
            ax = axes[row, col]
            theta = snapshots[direction][step]["theta"]
            lq, resp = mixture(x, theta[:2], theta[2:], weights)
            q = np.exp(lq)
            ax.plot(x, np.exp(lp), color="tab:blue", lw=1.6, label="p")
            ax.plot(x, q, "--", color="tab:orange", lw=1.8, label="q")
            ax.plot(x, q[:, None] * resp, color="tab:orange", lw=.7, alpha=.45)
            ax.text(.02, .94, f"iter {step}:  $\\mu$=({theta[0]:.2f}, {theta[1]:.2f}),  "
                    f"$\\sigma$=({theta[2]:.2f}, {theta[3]:.2f})",
                    transform=ax.transAxes, va="top", fontsize=9)
            ax.set_ylim(0, .45)
            ax.set_yticks([0, .2, .4])
            ax.set_xlim(-4.9, 14.9)
            ax.set_xticks(np.arange(-2.5, 13, 2.5))
            ax.grid(alpha=.25)
            if row == len(config["snapshots"]) - 1:
                ax.set_xlabel("x")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle(r"$a=10$, init $(\mu_1,\mu_2,\sigma_1,\sigma_2)=(0,a/4,1,1)$, "
                 "gradient descent lr=0.02", y=.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .975])
    fig.savefig(out / "reproduction.png", dpi=180)
    fig.savefig(out / "reproduction.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, direction in zip(axes, ("forward", "reverse")):
        rows = [r for r in traces if r["direction"] == direction]
        ax.plot([r["iteration"] for r in rows], [r["loss"] for r in rows])
        ax.set(xlabel="Gradient descent updates", ylabel=f"{direction.title()} KL (nats)",
               title=f"{direction.title()} objective")
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default="unversioned")
    parser.add_argument("--run-id", default="standalone")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    provenance = {
        "source_commit": args.source_commit, "run_id": args.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "pid": os.getpid(),
        "python": sys.version, "executable": sys.executable,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "config": config,
        "source_files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.iterdir()) if p.is_file()},
    }
    write_json(out / "PROVENANCE.json", provenance)
    obj = Objective(config)
    validation = {"finite_difference_checks": validate_gradients(obj)}
    write_json(out / "validation.json", validation)
    snapshots, traces = {}, []
    for direction in ("forward", "reverse"):
        snapshots[direction], rows = train(obj, config, direction)
        traces.extend(rows)
    save_csv(out / "trajectory.csv", traces)

    fine = Objective(config, refinement=2)
    refinement, comparison, summary = [], [], {}
    reference = json.loads((HERE / "reference_values.json").read_text())
    for direction in ("forward", "reverse"):
        for step, snapshot in snapshots[direction].items():
            theta = snapshot["theta"]
            loss, grad = obj.evaluate(theta, direction)
            fine_loss, fine_grad = fine.evaluate(theta, direction)
            refinement.append({"direction": direction, "iteration": step,
                               "loss_abs_difference": abs(loss-fine_loss),
                               "gradient_max_abs_difference": float(np.max(np.abs(grad-fine_grad)))})
            np.testing.assert_allclose(grad, fine_grad, rtol=1e-8, atol=1e-10)
            assert abs(loss-fine_loss) < 1e-10
            for parameter, value, expected in zip(
                    ["mu1", "mu2", "sigma1", "sigma2"], theta,
                    reference[direction][str(step)]):
                error = abs(float(value) - expected)
                comparison.append({"direction": direction, "iteration": step,
                                   "parameter": parameter, "computed": value,
                                   "image_annotation": expected, "absolute_error": error,
                                   "matches_image_rounding": error <= .005 + 1e-12})
        losses = np.array([r["loss"] for r in traces if r["direction"] == direction])
        final = snapshots[direction][config["steps"]]
        summary[direction] = {**final, "theta": final["theta"].tolist(),
                              "maximum_loss_increase": float(np.max(np.diff(losses))),
                              "loss_nonincreasing": bool(np.all(np.diff(losses) <= 1e-12))}
    save_csv(out / "reference_comparison.csv", comparison)
    validation["grid_refinement"] = refinement
    validation["all_image_annotations_match"] = all(r["matches_image_rounding"] for r in comparison)
    validation["matched_image_annotations"] = sum(r["matches_image_rounding"] for r in comparison)
    validation["total_image_annotations"] = len(comparison)
    validation["max_image_annotation_abs_error"] = max(r["absolute_error"] for r in comparison)
    write_json(out / "validation.json", validation)
    write_json(out / "snapshots.json", {
        direction: {step: {**s, "theta": s["theta"].tolist()} for step, s in rows.items()}
        for direction, rows in snapshots.items()})
    figures(config, snapshots, traces, out)
    summary["reference_match"] = {k: v for k, v in validation.items() if "image" in k}
    summary["elapsed_seconds"] = time.perf_counter() - start
    write_json(out / "summary.json", summary)
    provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["elapsed_seconds"] = summary["elapsed_seconds"]
    write_json(out / "PROVENANCE.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
