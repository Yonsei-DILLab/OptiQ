"""Validate the v5 mean-action OT contract without training or W&B."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_v3 import verify as verify_training

CONFIGS = {"mujoco_v5", "v5/final"}


def verify(overrides=(), config_name="mujoco_v5"):
    cfg = verify_training(overrides, config_name, allowed_configs=CONFIGS)
    actor = cfg.alg.actor
    checks = {
        "mean-action student OT": actor.get("ot_student_action") == "mean",
        "both mu-only evaluation modes": cfg.get("dual_mu_eval", False) and cfg.mu_only_eval,
        "no extra uniform collection": cfg.alg.behavior_uniform_probability == 0.,
        "fixed teacher temperature": not actor.get("temperature_schedule", {}).get("enabled", False),
    }
    for name, valid in checks.items():
        if not valid:
            raise ValueError(f"Not the canonical v5 algorithm: {name}")
    return cfg


if __name__ == "__main__":
    cfg = verify(sys.argv[1:], os.environ.get("OPTIQ_CONFIG", "mujoco_v5"))
    print(f"PASS: v5 {cfg.env_name}, seed={cfg.seed}, actor={cfg.alg.actor.hidden_dims}, "
          f"critic={cfg.alg.critic.hs}, T={cfg.alg.actor.temperature}, mean-action OT; "
          "Gaussian teacher/NLL, plain TD, dual mu-only eval, no uniform/annealing")
