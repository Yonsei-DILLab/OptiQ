"""Validate the v5 mean-action OT contract without training or W&B."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_v3 import verify as verify_training

PROFILES = {
    "mujoco_v5": (0., False), "v5/final": (0., False),
    "mujoco_v5_bestof8": (0., False), "v5/bestof8": (0., False),
    "mujoco_v5_behavior010": (.1, False), "v5/behavior010": (.1, False),
    "mujoco_v5_annealing": (0., True), "v5/annealing": (0., True),
    "mujoco_v5_annealing_behavior010": (.1, True), "v5/annealing_behavior010": (.1, True),
}
CONFIGS = set(PROFILES)


def verify(overrides=(), config_name="mujoco_v5"):
    winner = config_name in {"mujoco_v5_bestof8", "v5/bestof8"}
    cfg = verify_training(overrides, config_name, allowed_configs=CONFIGS,
        expected_teacher="best_of_k_winners" if winner else "conditional_mixture")
    actor = cfg.alg.actor
    probability, annealing = PROFILES[config_name]
    checks = {
        "mean-action student OT": actor.get("ot_student_action") == "mean",
        "both mu-only evaluation modes": cfg.get("dual_mu_eval", False) and cfg.mu_only_eval,
        "uniform collection probability matches profile": cfg.alg.behavior_uniform_probability == probability,
        "temperature schedule matches profile": actor.get("temperature_schedule", {}).get("enabled", False) == annealing,
        "best-of-k collection matches profile": cfg.alg.get("behavior_best_of_k", 1) == (
            8 if config_name in {"mujoco_v5_bestof8", "v5/bestof8"} else 1
        ),
    }
    for name, valid in checks.items():
        if not valid:
            raise ValueError(f"Not the requested v5 profile: {name}")
    return cfg


if __name__ == "__main__":
    cfg = verify(sys.argv[1:], os.environ.get("OPTIQ_CONFIG", "mujoco_v5"))
    print(f"PASS: v5 {cfg.env_name}, seed={cfg.seed}, actor={cfg.alg.actor.hidden_dims}, "
          f"critic={cfg.alg.critic.hs}, T={cfg.alg.actor.temperature}, mean-action OT; "
          f"teacher={cfg.alg.actor.teacher_distribution}, full NLL, plain TD, dual mu-only eval, "
          f"uniform p={cfg.alg.behavior_uniform_probability}, "
          f"schedule={cfg.alg.actor.get('temperature_schedule', None)}")
