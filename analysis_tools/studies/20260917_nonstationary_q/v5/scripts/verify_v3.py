"""Validate the actual v3 launch config without training, credentials or W&B."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from run_optiq_dime import validate_config


def verify(overrides=(), config_name="mujoco_v3", *, allowed_configs=None):
    if config_name not in (allowed_configs or {"mujoco_v3", "v3/final"}):
        raise ValueError("Use mujoco_v3 or v3/final for the v3 launcher")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    validate_config(cfg)
    actor, critic = cfg.alg.actor, cfg.alg.critic
    checks = {
        "plain scalar TD": critic.get("backup_mode") == "td" and critic.n_atoms == 1,
        "zero policy entropy coefficient": cfg.alg.ent_coef.init == 0.,
        "no soft guard": not actor.soft_guard.enabled,
        "no policy entropy evaluation": actor.entropy_samples == 0 and not actor.entropy_diagnostics,
        "continuous conditional Gaussian policy": actor.type == "semi_implicit" and actor.get("latent_prior", "normal") == "normal",
        "conditional-mixture teacher": actor.teacher_distribution == "conditional_mixture",
        "full OT NLL": actor.distillation_loss == "conditional_ot_nll",
        "no proximal acceptance substitute": actor.get("soft_proximal_ess_fraction", 0.) == 0.,
    }
    for name, valid in checks.items():
        if not valid:
            raise ValueError(f"Not the v3 algorithm: {name}")
    return cfg


if __name__ == "__main__":
    import os
    cfg = verify(sys.argv[1:], os.environ.get("OPTIQ_CONFIG", "mujoco_v3"))
    print(f"PASS: v3 {cfg.env_name}, seed={cfg.seed}, plain TD, no policy entropy/guard; "
          f"teacher T={cfg.alg.actor.temperature}, OT NLL.")
