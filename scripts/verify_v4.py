"""Compose v4 and check the inherited v3 training contract."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_v3 import verify as verify_training

CONFIGS = {"mujoco_v4", "v4/final", "mujoco_v4_behavior010", "v4/behavior010",
           "mujoco_v4_annealing", "v4/annealing"}


def verify(overrides=(), config_name="mujoco_v4"):
    if config_name not in CONFIGS:
        raise ValueError("Use a v4 configuration: " + ", ".join(sorted(CONFIGS)))
    cfg = verify_training(overrides, config_name, allowed_configs=CONFIGS)
    if not cfg.get("dual_mu_eval", False):
        raise ValueError("v4 requires both mu-only evaluation modes")
    return cfg


if __name__ == "__main__":
    cfg = verify(sys.argv[1:], os.environ.get("OPTIQ_CONFIG", "mujoco_v4"))
    print(f"PASS: v4 {cfg.env_name}, seed={cfg.seed}, actor={cfg.alg.actor.hidden_dims}, "
          f"critic={cfg.alg.critic.hs}, T={cfg.alg.actor.temperature}, "
          f"uniform collection p={cfg.alg.behavior_uniform_probability}, dual mu-only eval")
