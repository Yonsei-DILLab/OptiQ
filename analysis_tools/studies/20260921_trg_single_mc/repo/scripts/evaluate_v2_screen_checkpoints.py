"""Independent stochastic evaluations of final screen checkpoints, read-only.

The common predict/unscale path is used. Environment and policy RNG seeds here
are separate from screening and are shared across the fixed models. Confidence
intervals describe episode noise at these checkpoints, not training-seed variance.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flax import serialization
import gymnasium as gym
import jax
import numpy as np
from omegaconf import OmegaConf
from optiq_dime import OptiQDIME


def main():
    directories = [next((ROOT / "outputs/v2_screen").glob("*optiq-reference*"))]
    directories += sorted((ROOT / "outputs/v2_conditional_screen").glob("*T0.1-*"))
    results = []
    for directory in directories:
        cfg = OmegaConf.load(directory / "config.json")
        environment = gym.make("Humanoid-v4")
        model = OptiQDIME("MlpPolicy", environment, None, 1, cfg)
        checkpoint = next(directory.glob("checkpoints/*/actor_state_100000.msgpack"))
        model.policy.actor_state = serialization.from_bytes(model.policy.actor_state, checkpoint.read_bytes())
        episodes = []
        try:
            for index in range(30):
                env_seed, policy_seed = 910000 + index, 920000 + index
                obs, _ = environment.reset(seed=env_seed)
                model.policy.key = jax.random.PRNGKey(policy_seed)
                total, length, done = 0., 0, False
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, _ = environment.step(action)
                    total += float(reward)
                    length += 1
                    done = terminated or truncated
                episodes.append({"env_seed": env_seed, "policy_seed": policy_seed,
                    "return": total, "length": length})
            returns = np.array([e["return"] for e in episodes])
            result = {"run": directory.name, "training_seed": int(cfg.seed), "checkpoint_step": 100000,
                "episodes": episodes, "mean_return": float(returns.mean()),
                "return_std": float(returns.std(ddof=1)),
                "mean_length": float(np.mean([e["length"] for e in episodes]))}
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
        finally:
            model.get_env().close()
    reference = np.array([e["return"] for e in results[0]["episodes"]])
    rng = np.random.default_rng(930000)
    indices = rng.integers(0, len(reference), (10000, len(reference)))
    for result in results[1:]:
        differences = np.array([e["return"] for e in result["episodes"]]) - reference
        result["paired_gap_mean_vs_reference_seed0"] = float(differences.mean())
        result["episode_bootstrap_95_percent_interval"] = np.quantile(
            differences[indices].mean(axis=1), [.025, .975]).tolist()
    (ROOT / "outputs/v2_improvement/independent_100k_evaluation.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
