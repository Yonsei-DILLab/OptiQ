"""Read-only, independent stochastic evaluation of paired fixed checkpoints.

Every episode counts, including time-limit truncations. This secondary
checkpoint evaluation does not replace the predefined 900k--1M primary window.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENVIRONMENT_METRICS = ("reward_linvel", "reward_alive", "reward_quadctrl", "x_velocity")


def pool_environment_metrics(episodes):
    """Pool by recorded environment steps; short episodes get no extra weight."""
    pooled = {}
    for key in ENVIRONMENT_METRICS:
        values = [e["environment_metrics"][key] for e in episodes
                  if key in e.get("environment_metrics", {})]
        if values:
            total = sum(v["sum"] for v in values)
            count = sum(v["count"] for v in values)
            pooled[key] = {"sum": total, "count": count, "mean": total/count}
    return pooled


def evaluate_episodes(model, environment, count, seed_base):
    import jax
    episodes = []
    for index in range(count):
        env_seed, policy_seed = seed_base+index, seed_base+10000+index
        obs, _ = environment.reset(seed=env_seed)
        model.policy.key = jax.random.PRNGKey(policy_seed)
        total, length, done = 0., 0, False
        metric_sums, metric_counts = {}, {}
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = environment.step(action)
            total += float(reward)
            length += 1
            for key in ENVIRONMENT_METRICS:
                if key in info:
                    value = float(info[key])
                    if not np.isfinite(value):
                        raise ValueError(f"Nonfinite evaluation environment metric: {key}")
                    metric_sums[key] = metric_sums.get(key, 0.) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1
            done = terminated or truncated
        episodes.append({"env_seed": env_seed, "policy_seed": policy_seed,
                         "return": total, "length": length,
                         "terminated": bool(terminated), "truncated": bool(truncated),
                         "environment_metrics": {
                             key: {"sum": value, "count": metric_counts[key],
                                   "mean": value/metric_counts[key]}
                             for key, value in metric_sums.items()}})
    return episodes


def paired_summary(results, seed_base):
    grouped = {(r["training_seed"], r["method"]): r for r in results}
    if set(grouped) != {(s, m) for s in range(4) for m in ("v2", "optiq")}:
        return {"complete": False}
    rng = np.random.default_rng(seed_base+20000)
    pairs = []
    for seed in range(4):
        a, b = grouped[seed, "v2"], grouped[seed, "optiq"]
        assert [(e["env_seed"], e["policy_seed"]) for e in a["episodes"]] == [
            (e["env_seed"], e["policy_seed"]) for e in b["episodes"]]
        differences = np.array([e["return"] for e in a["episodes"]]) - np.array([e["return"] for e in b["episodes"]])
        indices = rng.integers(0, len(differences), (10000, len(differences)))
        pairs.append({"training_seed": seed, "v2_mean": a["mean_return"], "optiq_mean": b["mean_return"],
                      "paired_gap_mean": float(differences.mean()),
                      "episode_bootstrap_95_percent_interval": np.quantile(differences[indices].mean(axis=1), [.025, .975]).tolist()})
    means = {m: [grouped[s, m]["mean_return"] for s in range(4)] for m in ("v2", "optiq")}
    return {"complete": True, "pairs": pairs,
            "methods": {m: {"mean": float(np.mean(v)), "seed_sd": float(np.std(v, ddof=1)), "seed_scores": v}
                        for m, v in means.items()},
            "interval_scope": "Episode uncertainty conditional on each fixed checkpoint pair; not training-seed uncertainty."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/v2_improvement/confirmation_manifest.json")
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=970000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 2 or args.step < 1:
        parser.error("At least two episodes and a positive checkpoint step are required")
    from flax import serialization
    import gymnasium as gym
    import jax
    from omegaconf import OmegaConf
    from optiq_dime import OptiQDIME

    manifest = json.loads(args.manifest.read_text())
    assert len(manifest["runs"]) == 8
    assert {(r["seed"], r["method"]) for r in manifest["runs"]} == {(s, m) for s in range(4) for m in ("v2", "optiq")}
    output = args.output or ROOT / f"outputs/v2_improvement/confirmation_independent_{args.step:07d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_step": args.step,
              "episodes_per_run": args.episodes, "seed_base": args.seed_base,
              "jax_devices": [str(d) for d in jax.devices()], "jax_version": jax.__version__,
              "gymnasium_version": gym.__version__, "results": [], "summary": {"complete": False}}
    for item in sorted(manifest["runs"], key=lambda r: (r["seed"], r["method"])):
        directory = Path(item["directory"])
        cfg = OmegaConf.load(directory / "config.json")
        assert cfg.env_name == "Humanoid-v4" and cfg.seed == item["seed"]
        assert cfg.runtime.git_commit == item["commit"]
        checkpoint = next(directory.glob(f"checkpoints/*/actor_state_{args.step}.msgpack"))
        payload = checkpoint.read_bytes()
        environment = gym.make(cfg.env_name)
        model = OptiQDIME("MlpPolicy", environment, None, 1, cfg)
        try:
            model.policy.actor_state = serialization.from_bytes(model.policy.actor_state, payload)
            episodes = evaluate_episodes(model, environment, args.episodes, args.seed_base)
            returns = np.array([e["return"] for e in episodes])
            result = {"run": directory.name, "method": item["method"], "training_seed": int(cfg.seed),
                      "source_commit": item["commit"], "checkpoint": str(checkpoint),
                      "checkpoint_sha256": hashlib.sha256(payload).hexdigest(), "episodes": episodes,
                      "mean_return": float(returns.mean()), "episode_return_sd": float(returns.std(ddof=1)),
                      "mean_length": float(np.mean([e["length"] for e in episodes])),
                      "environment_metrics": pool_environment_metrics(episodes),
                      "time_limit_episodes": sum(e["truncated"] and not e["terminated"] for e in episodes)}
            record["results"].append(result)
            record["summary"] = paired_summary(record["results"], args.seed_base)
            record["updated_utc"] = datetime.now(timezone.utc).isoformat()
            output.write_text(json.dumps(record, indent=2)+"\n")
            print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
        finally:
            model.get_env().close()
    print(json.dumps(record["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
