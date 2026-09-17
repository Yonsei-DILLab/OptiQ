"""Reevaluate a fixed v2 seed before/at an observed evaluation dip.

This is a post-hoc checkpoint diagnostic, not the predefined final comparison.
It uses new, shared episodes and the unchanged common stochastic evaluator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=range(4))
    parser.add_argument("--steps", required=True, type=int, nargs="+")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-base", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.episodes < 2 or any(s < 1 for s in args.steps):
        parser.error("At least two episodes and positive checkpoint steps are required")
    from flax import serialization
    import gymnasium as gym
    import jax
    import numpy as np
    from omegaconf import OmegaConf
    from optiq_dime import OptiQDIME
    from scripts.evaluate_v2_confirmation import evaluate_episodes, pool_environment_metrics

    manifest = json.loads((ROOT / "outputs/v2_improvement/confirmation_manifest.json").read_text())
    items = [r for r in manifest["runs"] if r["method"] == "v2" and r["seed"] == args.seed]
    assert len(items) == 1
    item = items[0]
    directory = Path(item["directory"])
    cfg = OmegaConf.load(directory / "config.json")
    assert cfg.seed == args.seed and cfg.env_name == "Humanoid-v4"
    assert cfg.runtime.git_commit == item["commit"]
    assert cfg.alg.behavior_uniform_probability == 0
    checkpoints = {step: next(directory.glob(f"checkpoints/*/actor_state_{step}.msgpack"))
                   for step in sorted(set(args.steps))}
    with np.load(next(directory.glob("eval/*/evaluations.npz"))) as data:
        eval_steps, eval_returns, eval_lengths = [data[k].copy() for k in ("timesteps", "results", "ep_lengths")]
    record = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "training_seed": args.seed, "steps": list(checkpoints),
              "episodes_per_checkpoint": args.episodes, "seed_base": args.seed_base,
              "source_commit": item["commit"],
              "evaluation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "common_evaluator_sha256": hashlib.sha256((ROOT / "scripts/evaluate_v2_confirmation.py").read_bytes()).hexdigest(),
              "jax_version": jax.__version__, "gymnasium_version": gym.__version__,
              "devices": [str(d) for d in jax.devices()], "results": [], "complete": False,
              "scope": "Post-hoc fixed-checkpoint diagnostic of one training seed; not a final performance test or causal diagnosis."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step, checkpoint in checkpoints.items():
        payload = checkpoint.read_bytes()
        environment = gym.make(cfg.env_name)
        model = OptiQDIME("MlpPolicy", environment, None, 1, cfg)
        try:
            model.policy.actor_state = serialization.from_bytes(model.policy.actor_state, payload)
            episodes = evaluate_episodes(model, environment, args.episodes, args.seed_base)
            returns = np.array([e["return"] for e in episodes])
            assert len(episodes) == args.episodes and np.isfinite(returns).all()
            index = np.flatnonzero(eval_steps == step)
            assert len(index) == 1
            result = {"step": step, "checkpoint": str(checkpoint),
                      "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                      "original_evaluation_mean": float(eval_returns[index[0]].mean()),
                      "original_evaluation_mean_length": float(eval_lengths[index[0]].mean()),
                      "episodes": episodes, "mean_return": float(returns.mean()),
                      "episode_return_sd": float(returns.std(ddof=1)),
                      "mean_length": float(np.mean([e["length"] for e in episodes])),
                      "time_limit_episodes": sum(e["truncated"] and not e["terminated"] for e in episodes),
                      "environment_metrics": pool_environment_metrics(episodes)}
            record["results"].append(result)
            record["complete"] = len(record["results"]) == len(checkpoints)
            record["updated_utc"] = datetime.now(timezone.utc).isoformat()
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
        finally:
            model.get_env().close()


if __name__ == "__main__":
    main()
