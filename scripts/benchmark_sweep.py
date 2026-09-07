"""Bounded, offline timing probe; run only on an idle, healthy GPU.

Examples (repository root):
  python scripts/benchmark_sweep.py --host vast2 --gpu 2 --task dog-run
  python scripts/benchmark_sweep.py --host vast2 --gpu 2 --task dog-run --no-anchor
  python scripts/benchmark_sweep.py --cpu-smoke
Outputs go to a unique temporary directory unless --output is supplied.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["vast1", "vast2"])
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--task", default="dog-run")
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument("--diagnostics-interval", type=int, default=1000)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=1100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.updates < 1 or args.burn_in < 0:
        parser.error("updates must be positive; burn-in must be nonnegative")
    lock = None
    if args.cpu_smoke:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args.updates, args.burn_in = 10, 6
    else:
        if args.host is None or args.gpu is None:
            parser.error("--host and --gpu required; no automatic GPU allocation")
        if args.host == "vast1" and args.gpu == 1:
            parser.error("vast1 GPU 1 excluded due to repeated historical CUDA errors")
        lock_path = Path("/workspace/optiq-beta-sweep-20260905/locks") / f"{args.host}_gpu{args.gpu}.lock"
        lock = lock_path.open("a")
        # Existing queue workers hold this lock across queued jobs, including
        # their short process-to-process idle gaps. Never bypass that queue.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([
            "nvidia-smi", f"--id={args.gpu}", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True)
        if any(line.strip().isdigit() for line in result.stdout.splitlines()):
            raise RuntimeError("GPU is busy; no training job has been stopped")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["WANDB_MODE"] = "disabled"
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import jax
    import numpy as np
    from hydra import compose, initialize_config_dir
    from stable_baselines3.common.evaluation import evaluate_policy
    from scripts.parameter_sweep import DEFAULT_SPEC, build_runs
    from run_optiq_dime import create_algorithm

    run = next(build_runs(json.loads(DEFAULT_SPEC.read_text()), [args.task], [0]))
    output = args.output or Path(tempfile.mkdtemp(prefix="optiq-timing-"))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "timing.json").exists():
        raise FileExistsError("Refusing to overwrite an existing timing result")
    warmup = 8 if args.cpu_smoke else 5000
    diagnostic_interval = 4 if args.cpu_smoke else args.diagnostics_interval
    overrides = run["argv"][3:] + [
        "wandb.activate=false", "eval_interval=0", "checkpoint_interval=0",
        f"diagnostics_interval={diagnostic_interval}",
        f"output_root={output}", f"alg.learning_starts={warmup}",
        "alg.actor.proposal_std=0.2", "alg.actor.proposal_clip=0.5",
        "alg.actor.temperature=0.25", "alg.actor.density_beta=0.1",
        f"alg.actor.include_anchor={str(not args.no_anchor).lower()}",
        f"alg.actor.proposals_per_policy_sample={4 if args.no_anchor else 5}",
    ]
    if args.cpu_smoke:
        overrides += ["env_name=Pendulum-v1", "alg.critic.hs=[16,16]",
                      "alg.actor.hidden_dims=[16,16]", "alg.critic.n_atoms=11",
                      "alg.batch_size=2", "alg.buffer_size=64",
                      "alg.actor.num_policy_samples=4"]
    with initialize_config_dir(version_base=None, config_dir=str(DEFAULT_SPEC.parents[1])):
        cfg = compose(config_name="parameter_sweep", overrides=overrides)
    model, callbacks = create_algorithm(cfg)
    training_latencies, stamps = [], []
    original_train = model.train

    def timed_train(*a, **kw):
        start = time.perf_counter()
        original_train(*a, **kw)
        jax.block_until_ready((model.policy.qf_state.params, model.policy.actor_state.params, model.key))
        end = time.perf_counter()
        training_latencies.append(end - start)
        stamps.append(end)

    model.train = timed_train
    try:
        start = time.perf_counter()
        model.learn(total_timesteps=warmup + args.burn_in + args.updates,
                    callback=callbacks, progress_bar=False)
        elapsed = time.perf_counter() - start
        # End-to-end stable iteration gaps include environment, actor action
        # generation, replay and optimizer work, but exclude initial JITs.
        stable = np.diff(stamps)[args.burn_in:]
        n_eval = 1 if args.cpu_smoke else 10
        eval_env = callbacks.callbacks[0].eval_env
        eval_start = time.perf_counter()
        rewards, lengths = evaluate_policy(model, eval_env, n_eval_episodes=n_eval,
                                          deterministic=False, return_episode_rewards=True)
        eval_seconds = time.perf_counter() - eval_start
        model.model_save_path = str(output / "checkpoints")
        Path(model.model_save_path).mkdir(exist_ok=True)
        checkpoint_start = time.perf_counter()
        model._save_model()
        checkpoint_seconds = time.perf_counter() - checkpoint_start
        training_seconds = float(stable.mean()) * (1_000_000 - 5000)
        estimate = training_seconds + 101 * eval_seconds * 10 / n_eval + 21 * checkpoint_seconds
        report = {
            "cpu_smoke_only": args.cpu_smoke, "task": args.task,
            "env_name": cfg.env_name, "anchor": not args.no_anchor,
            "diagnostics_interval": diagnostic_interval,
            "device": str(jax.devices()[0]), "jax_version": jax.__version__,
            "training_steps_measured": len(stable), "probe_wall_seconds": elapsed,
            "steady_step_ms": float(stable.mean()) * 1000,
            "step_ms_p50": float(np.median(stable)) * 1000,
            "step_ms_p95": float(np.quantile(stable, .95)) * 1000,
            "eval_seconds": eval_seconds, "eval_episodes": n_eval,
            "eval_mean_length": float(np.mean(lengths)),
            "checkpoint_seconds": checkpoint_seconds,
            "estimated_1m_hours_excluding_startup": estimate / 3600,
            "caveats": "Early-training single-GPU synchronous estimate, not a completed 1M run. Add setup, host variation and failures. Never use CPU smoke numbers for capacity planning.",
        }
        (output / "timing.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        print(f"RESULT: {output / 'timing.json'}", flush=True)
    finally:
        model.env.close()
        callbacks.callbacks[0].eval_env.close()
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
