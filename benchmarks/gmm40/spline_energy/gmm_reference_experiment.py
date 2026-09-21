"""Five-seed GMM40 check of the restored raw-energy implementation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithms.spline_energy.model import log_prob_from_output, q_from_output
from algorithms.spline_energy.raw_energy import (
    StateFreeRawEnergyCircuit, fixed_q_forward_energy_loss,
    sample_many_from_single_output)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--target-data", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=["offline", "disabled"], default="offline")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.seed not in range(5) or args.steps != 100_000:
        raise ValueError("The committed comparison is exactly seeds 0..4 and 100k updates")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit the exact source before training")
    upstream_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip()
    if upstream_commit != "811e0e2e59a0c9137f38433e4fc18adaf5e6a8fa":
        raise RuntimeError(f"Unexpected frozen target/evaluation source: {upstream_commit}")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["GMM40_REPO_ROOT"] = str(args.upstream)
    os.environ["GMM40_RESULTS_ROOT"] = str(args.target_data)
    sys.path.insert(0, str(args.upstream))
    from gmm40.target import Target
    from gmm40.evaluation import metrics as reference_metrics

    target = Target()
    reference = target.sample(10_000, 20260921, bounded=True)
    full_reference = target.sample(10_000, 20260921, bounded=False)
    model = StateFreeRawEnergyCircuit(initialization_seed=args.seed)
    params = model.init(jax.random.PRNGKey(args.seed))["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(3e-4))
    key = jax.random.PRNGKey(args.seed)

    @jax.jit
    def update(carry, _):
        train_state, rng = carry
        rng, sample_key, uniform_key, mixture_key = jax.random.split(rng, 4)
        output = model.apply({"params": train_state.params})
        policy = sample_many_from_single_output(output, sample_key, 16_384)
        uniform = jax.random.uniform(uniform_key, (16_384, 2), minval=-1., maxval=1.)
        actions = jax.lax.stop_gradient(jnp.where(
            (jax.random.uniform(mixture_key, (16_384,)) < .5)[:, None], uniform, policy))
        proposal_log_prob = jnp.logaddexp(
            log_prob_from_output(output, actions) - math.log(2.), -math.log(8.))
        target_log_energy = target.jax_log_prob(40. * actions) + 2. * math.log(40.) - target.log_z

        def loss_fn(candidate):
            candidate_output = model.apply({"params": candidate})
            return fixed_q_forward_energy_loss(
                candidate_output, actions, target_log_energy, proposal_log_prob)

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        info = {"loss": loss, "grad_norm": optax.global_norm(grads),
                "target_integral_estimate": jnp.mean(jnp.exp(target_log_energy - proposal_log_prob))}
        return (train_state, rng), info

    @jax.jit
    def advance(train_state, rng):
        (train_state, rng), info = jax.lax.scan(update, (train_state, rng), None, length=5_000)
        return train_state, rng, jax.tree_util.tree_map(jnp.mean, info)

    def evaluate(train_state):
        output = model.apply({"params": train_state.params})
        normalized = np.asarray(sample_many_from_single_output(
            output, jax.random.PRNGKey(9102026), 10_000))
        physical = 40. * normalized
        result = reference_metrics(physical, target, reference, full_reference)
        target_normalized = jnp.asarray(reference / 40.)
        sample_normalized = jnp.asarray(normalized)
        target_lp = target.log_prob(reference, bounded=True) + 2. * math.log(40.)
        sample_lp = target.log_prob(physical, bounded=True) + 2. * math.log(40.)
        result.update(
            forward_kl=float(np.mean(target_lp - np.asarray(log_prob_from_output(output, target_normalized)))),
            reverse_kl=float(np.mean(np.asarray(log_prob_from_output(output, sample_normalized)) - sample_lp)),
            log_partition=float(np.asarray(output["log_partition"])[0]),
        )
        return result, physical

    config = {
        "algorithm": "spline_energy_gmm_reference", "seed": args.seed,
        "steps": args.steps, "rank": 64, "knots": 129, "temperature": 1.,
        "learning_rate": 3e-4, "target_queries_per_update": 16_384,
        "defensive_uniform_probability": .5, "eval_every": 5_000,
        "eval_samples": 10_000, "source_commit": source_commit,
        "upstream_commit": upstream_commit, "trainable_parameters": sum(
            x.size for x in jax.tree_util.tree_leaves(params)),
        "changed_variable": "restored StateFreeRawEnergyCircuit implementation only",
        "baseline": "preserved five-seed successful GMM40 spline campaign",
    }
    metadata = {
        "config": config, "command": sys.argv, "hostname": platform.node(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "device": str(jax.devices()),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                                         "--format=csv,noheader"], text=True).strip(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["jax", "jaxlib", "flax", "optax", "numpy", "scipy", "wandb"]},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output / "manifest.json", metadata)
    run = wandb.init(project="OptiQ-GMM40-Spline-Reference", group="gmm_reference_100k",
                     name=f"GMM40-gmm-reference-s{args.seed}", mode=args.wandb_mode,
                     dir=str(args.output), config=config,
                     id=hashlib.sha256(str(args.output).encode()).hexdigest()[:12], resume="never")
    (args.output / "wandb.json").write_text(json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n")
    run.define_metric("update"); run.define_metric("*", step_metric="update")
    # Compile and discard: it must not consume a training update or RNG state.
    compiled = advance.lower(state, key).compile()
    training_seconds = 0.
    with (args.output / "history.jsonl").open("w") as history:
        for step in range(0, args.steps + 1, 5_000):
            info = {}
            if step:
                begin = time.monotonic()
                state, key, info = compiled(state, key)
                jax.block_until_ready(state.params)
                training_seconds += time.monotonic() - begin
            if not finite((state.params, info)) or int(state.step) != step:
                raise FloatingPointError(f"Invalid state at update {step}")
            scores, samples = evaluate(state)
            row = {"update": step, "training_seconds": training_seconds,
                   "target_queries": step * 16_384, **scores,
                   **{f"train/{name}": float(value) for name, value in info.items()}}
            history.write(json.dumps(row, allow_nan=False) + "\n"); history.flush()
            atomic_json(args.output / "latest.json", row)
            run.log({k: v for k, v in row.items() if isinstance(v, (int, float))}, step=step)
            print(json.dumps({"seed": args.seed, "update": step,
                "modes": scores["mode_coverage"], "three_sigma": scores["high_density_fraction"],
                "forward_kl": scores["forward_kl"], "reverse_kl": scores["reverse_kl"]}), flush=True)
    (args.output / "checkpoint_100000.msgpack").write_bytes(
        serialization.to_bytes({"state": state, "key": key}))
    np.save(args.output / "samples_100000.npy", samples)
    complete = {"completed": True, "seed": args.seed, "steps": args.steps,
                "training_seconds": training_seconds, "metrics": row,
                "finished_utc": datetime.now(timezone.utc).isoformat(), "wandb_url": run.url}
    atomic_json(args.output / "COMPLETE.json", complete)
    run.summary.update(complete); run.finish()


if __name__ == "__main__":
    main()
