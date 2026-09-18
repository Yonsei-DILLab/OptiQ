"""Evaluate the historical 10% behavior reference on final comparison episodes.

This is a secondary baseline, kept separate from the matched zero-uniform
confirmation. Evaluation uses the ordinary stochastic policy, without extra
uniform behavior. Existing models and training processes are never changed.
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

from flax import serialization
import gymnasium as gym
import jax
import numpy as np
from omegaconf import OmegaConf

from optiq_dime import OptiQDIME
from scripts.evaluate_v2_confirmation import evaluate_episodes, pool_environment_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "outputs/v2_improvement/historical_behavior010_reference.json")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1100000)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/v2_improvement/historical_behavior010_independent_1000000.json")
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("At least two episodes are required")
    reference = json.loads(args.reference.read_text())
    items = sorted(reference["runs"], key=lambda r: r["seed"])
    assert [r["seed"] for r in items] == [0, 1, 2, 3]
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "step": 1000000,
              "episodes_per_run": args.episodes, "seed_base": args.seed_base,
              "method": "historical_optiq_behavior010", "results": [], "complete": False,
              "evaluation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "common_evaluator_sha256": hashlib.sha256((ROOT / "scripts/evaluate_v2_confirmation.py").read_bytes()).hexdigest(),
              "jax_version": jax.__version__, "gymnasium_version": gym.__version__,
              "devices": [str(d) for d in jax.devices()],
              "scope": "Different behavior-collection setting; separate historical reference, not the primary zero-uniform control."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in items:
        directory = Path(item["directory"])
        cfg = OmegaConf.load(directory / "config.json")
        assert cfg.seed == item["seed"] and cfg.env_name == "Humanoid-v4"
        assert cfg.alg.behavior_uniform_probability == .1 and cfg.alg.actor.get("type", "implicit") == "implicit"
        assert (directory / "completed.json").exists()
        checkpoint = next(directory.glob("checkpoints/*/actor_state_1000000.msgpack"))
        payload = checkpoint.read_bytes()
        environment = gym.make(cfg.env_name)
        model = OptiQDIME("MlpPolicy", environment, None, 1, cfg)
        try:
            model.policy.actor_state = serialization.from_bytes(model.policy.actor_state, payload)
            episodes = evaluate_episodes(model, environment, args.episodes, args.seed_base)
            returns = np.array([e["return"] for e in episodes])
            assert len(episodes) == args.episodes and np.isfinite(returns).all()
            result = {"seed": item["seed"], "run": directory.name,
                      "recorded_git_metadata": item.get("recorded_git_metadata", {}),
                      "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                      "episodes": episodes, "mean_return": float(returns.mean()),
                      "episode_return_sd": float(returns.std(ddof=1)),
                      "mean_length": float(np.mean([e["length"] for e in episodes])),
                      "time_limit_episodes": sum(e["truncated"] and not e["terminated"] for e in episodes),
                      "environment_metrics": pool_environment_metrics(episodes)}
            record["results"].append(result)
            record["complete"] = len(record["results"]) == 4
            record["updated_utc"] = datetime.now(timezone.utc).isoformat()
            args.output.write_text(json.dumps(record, indent=2)+"\n")
            print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
        finally:
            model.get_env().close()


if __name__ == "__main__":
    main()
