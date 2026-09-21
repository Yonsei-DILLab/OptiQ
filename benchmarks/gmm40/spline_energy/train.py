"""Reproducible GMM40 experiment; online W&B, local results, immutable attempts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from flax import serialization
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from core import (defaults, initialize, make_update, mixture_log_prob,
                  mixture_sample, target_logp, target_parameters)
from smem_tr import DEFAULTS as SMEM_DEFAULTS, add_arguments, validate_config

ROOT = Path(__file__).resolve().parent
PROJECT = "OptiQ-GMM40-Sampling-Comparison"


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def code_state():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("Commit exact source before training: "+dirty)
    return sha


def labels(x, centers):
    return np.argmin(((x[:, None]-centers[None])**2).sum(-1), axis=1)


def evaluation_set(target, count):
    x = mixture_sample(jax.random.PRNGKey(20260921), *target, (count,))
    logp = target_logp(x, target)
    centers = np.asarray(target[0])
    refmass = np.bincount(labels(np.asarray(x), centers), minlength=40)/count
    return x, logp, refmass


@jax.jit
def evaluate_arrays(mu, ls, target, tx):
    x = mixture_sample(jax.random.PRNGKey(9102026), mu, ls, (tx.shape[0],))
    lq_target = mixture_log_prob(tx[None], mu[None], ls[None])[0]
    lq = mixture_log_prob(x[None], mu[None], ls[None])[0]
    lp = target_logp(x, target)
    return x, lq_target, lq, lp


def evaluate(state, z, target, evaluation):
    mu, ls = state.apply_fn({"params": state.params}, z)
    tx, tlp, refmass = evaluation
    x, tlq, lq, lp = evaluate_arrays(mu, ls, target, tx)
    x, tlq, lq, lp, tlp = map(np.asarray, (x, tlq, lq, lp, tlp))
    centers, std = np.asarray(target[0]), np.exp(np.asarray(target[1]))
    assignment = labels(x, centers)
    mass = np.bincount(assignment, minlength=40)/len(x)
    coverage = int(np.sum(mass >= .1*refmass))
    distances = (((x[:, None]-centers[None])/std[None])**2).sum(-1)
    near = distances.min(-1) < 9.
    near_mass = np.bincount(assignment, weights=near.astype(float), minlength=40)/len(x)
    metrics = dict(forward_kl=float(np.mean(tlp-tlq)), reverse_kl=float(np.mean(lq-lp)),
        target_nll=float(-tlq.mean()), mode_coverage=coverage, mode_mass_tv=float(.5*np.abs(mass-refmass).sum()),
        precision_3sigma=float(np.mean(near)), mode_coverage_3sigma=int(np.sum(near_mass >= .1*refmass)),
        min_mode_mass=float(mass.min()), max_mode_mass=float(mass.max()))
    if not all(np.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite evaluation")
    return metrics, x, np.asarray(mu), np.asarray(ls), mass, refmass


def plot_distribution(path, target, samples, mu, mass, refmass, title):
    centers = np.asarray(target[0])*50.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axs[0].scatter(*(samples[:6000]*50.).T, s=2, alpha=.18, c="#2171b5", rasterized=True)
    axs[0].scatter(*centers.T, marker="x", s=30, c="#d95f02", label="Target modes")
    axs[0].scatter(*(mu*50.).T, marker="+", s=35, c="#31a354", label="GMM component centers")
    axs[0].set(xlim=(-50, 50), ylim=(-50, 50), xlabel="x1", ylabel="x2", aspect="equal")
    axs[0].legend(fontsize=8, loc="upper right")
    ids = np.arange(40)
    axs[1].bar(ids-.2, refmass, width=.4, color="#d95f02", label="Target cell mass")
    axs[1].bar(ids+.2, mass, width=.4, color="#2171b5", label="Actor cell mass")
    axs[1].set(xlabel="Nearest target mode", ylabel="Probability", ylim=(0, max(.06, mass.max()*1.1)))
    axs[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["snis", "smc_mala", "snis_smem_tr",
                                        "snis_smem_tr_aux"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--phase", default="main")
    p.add_argument("--attempt", default="a1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--defensive", type=float, default=.1)
    p.add_argument("--stages", type=int, default=16)
    p.add_argument("--moves", type=int, default=2)
    p.add_argument("--mala-step", type=float, default=.2)
    add_arguments(p)
    args = p.parse_args()
    cfg = defaults(args.method, args.seed)
    cfg.update(steps=args.steps, phase=args.phase, eval_every=args.eval_every,
               defensive=args.defensive, stages=args.stages, moves=args.moves, mala_step=args.mala_step)
    if cfg["stages"] < 1 or cfg["moves"] < 1:
        raise ValueError("Stages and moves must be positive")
    if args.method in ("snis_smem_tr", "snis_smem_tr_aux"):
        cfg.update({name: getattr(args, name) for name in SMEM_DEFAULTS})
        validate_config(cfg)
    sha = code_state()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    devices = [str(x) for x in jax.devices()]
    metadata = dict(config=cfg, source_commit=sha,
        upstream_commit="4ca69473515b5083d6e651845ded39da34e14538",
        command=sys.argv, hostname=platform.node(), device=devices,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
        output=str(out), packages={n: importlib.metadata.version(n) for n in ["jax", "jaxlib", "flax", "optax", "wandb", "numpy"]})
    try:
        metadata["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["gpu"] = None
    state, z, key = initialize(args.seed, cfg["components"], cfg["lr"])
    target = target_parameters()
    np.savez(out/"target.npz", centers=np.asarray(target[0]), log_std=np.asarray(target[1]))
    cfg["trainable_parameters"] = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    write_json(out/"manifest.json", metadata)
    evaluation = evaluation_set(target, cfg["eval_samples"])
    run_id = hashlib.sha256(str(out).encode()).hexdigest()[:12]
    method_name = {"snis": "DirectGMM-SNIS", "smc_mala": "DirectGMM-SMC-MALA",
                   "snis_smem_tr": "DirectGMM-SNIS-SMEM-TR",
                   "snis_smem_tr_aux": "DirectGMM-SNIS-SMEM-TR-Aux"}[args.method]
    name = f"GMM40-{method_name}-{args.phase}-seed{args.seed}-{args.attempt}"
    run = wandb.init(entity="OptiQ", project=PROJECT, name=name, id=run_id, resume="never",
        group=f"{args.phase}-{args.method}", job_type="train", dir=str(out),
        config=dict(cfg, source_commit=sha, slurm_job_id=metadata["slurm_job_id"], devices=devices),
        tags=["GMM40", "Direct-GMM-NLL", args.method, args.phase],
        settings=wandb.Settings(init_timeout=120))
    write_json(out/"wandb.json", dict(id=run.id, url=run.url, project=PROJECT))
    run.define_metric("update")
    run.define_metric("*", step_metric="update")
    run.summary.update(metadata)
    source_artifact = wandb.Artifact("gmm40-source-"+sha[:12], type="code", metadata={"commit": sha})
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path.suffix in (".py", ".md", ".sh"):
            source_artifact.add_file(str(path), name=str(path.relative_to(ROOT)))
    run.log_artifact(source_artifact)
    updater = make_update(target, cfg)
    # Compilation is not a training update: discard the result and keep original key/state.
    start = time.monotonic()
    warm = updater(state, z, key)
    jax.block_until_ready(warm)
    compilation_seconds = time.monotonic()-start
    run.summary["compilation_seconds"] = compilation_seconds
    print(json.dumps(dict(event="ready", name=name, url=run.url, compile_seconds=compilation_seconds)), flush=True)
    training_seconds = 0.
    start = time.monotonic()
    info = {}
    info_window = []
    try:
        with (out/"history.jsonl").open("a") as history:
            for step in range(args.steps+1):
                if step:
                    tick = time.monotonic()
                    state, key, info = updater(state, z, key)
                    jax.block_until_ready(state)
                    training_seconds += time.monotonic()-tick
                    if args.method in ("snis_smem_tr", "snis_smem_tr_aux"):
                        info_window.append(info)
                if step % args.eval_every and step != args.steps:
                    continue
                metrics, samples, mu, ls, mass, refmass = evaluate(state, z, target, evaluation)
                finite = all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
                if not finite:
                    raise FloatingPointError("Nonfinite state")
                scalar_info = {k:float(v) for k,v in info.items()}
                if info_window:
                    # Structural searches usually fall between evaluations;
                    # preserve interval counts instead of logging only the last
                    # (typically non-structural) update's zero indicator.
                    for metric_name in ("smem_attempted", "smem_selected", "actor_accepted", "final_actor_rejected"):
                        scalar_info[metric_name+"_interval_count"] = sum(float(i[metric_name]) for i in info_window)
                    for metric_name in ("teacher_nll_gain", "actor_nll_gain", "projection_loss"):
                        scalar_info[metric_name+"_interval_mean"] = float(np.mean([float(i[metric_name]) for i in info_window]))
                    for metric_name in ("teacher_kl_bound", "actor_kl_bound"):
                        scalar_info[metric_name+"_interval_max"] = max(float(i[metric_name]) for i in info_window)
                    info_window.clear()
                if not all(np.isfinite(v) for v in scalar_info.values()):
                    raise FloatingPointError("Nonfinite sampler/update statistic")
                # Per-point target log densities; every score is separately counted.
                n = cfg["particles"]*cfg["populations"]
                per_update = n*(1+cfg["stages"]*(1+2*cfg["moves"])) if args.method == "smc_mala" else n
                gradient_per_update = n*cfg["stages"]*2*cfg["moves"] if args.method == "smc_mala" else 0
                row = dict(update=step, wall_seconds=time.monotonic()-start,
                    training_seconds=training_seconds, target_logp_evals=step*per_update,
                    target_score_evals=step*gradient_per_update, **metrics, **scalar_info)
                history.write(json.dumps(row, allow_nan=False)+"\n"); history.flush()
                write_json(out/"latest.json", row)
                plot_path = out/f"distribution_{step:06d}.png"
                plot_distribution(plot_path, target, samples, mu, mass, refmass,
                    f"{name} | update {step} | KL(p||q)={metrics['forward_kl']:.3f} | modes {metrics['mode_coverage']}/40")
                run.log(dict(row, distribution=wandb.Image(str(plot_path))), step=step)
                run.summary.update(row)
                if step == 0 or step % 1000 == 0 or step == args.steps:
                    ckpt = out/f"checkpoint_{step:06d}.msgpack"
                    ckpt.write_bytes(serialization.to_bytes(dict(state=state, z=z, key=key)))
                    np.savez_compressed(out/f"samples_{step:06d}.npz", samples=samples, mu=mu,
                        log_std=ls, mode_mass=mass, reference_mass=refmass)
                print(json.dumps(row, allow_nan=False), flush=True)
        artifact = wandb.Artifact(name.lower(), type="gmm40-result", metadata={"source_commit":sha, "seed":args.seed, "method":args.method})
        for file in [out/"manifest.json", out/"history.jsonl", out/"latest.json", out/"target.npz",
                     out/f"checkpoint_{args.steps:06d}.msgpack", out/f"samples_{args.steps:06d}.npz",
                     out/f"distribution_{args.steps:06d}.png"]:
            artifact.add_file(str(file))
        run.log_artifact(artifact)
        run.summary["completed"] = True
        run.finish()
        write_json(out/"COMPLETE.json", dict(steps=args.steps, source_commit=sha, wandb_url=run.url, metrics=row))
    except BaseException as exc:
        write_json(out/"FAILED.json", dict(error=repr(exc), source_commit=sha))
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
