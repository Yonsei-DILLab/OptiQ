"""Validate the v7 launch contract without training or starting W&B."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from run_optiq_dime import validate_config

CONFIGS = {"mujoco_v7", "v7/final"}


def verify(overrides=(), config_name="mujoco_v7"):
    if config_name not in CONFIGS:
        raise ValueError("Use mujoco_v7 or v7/final for the v7 launcher")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    validate_config(cfg)
    actor, critic = cfg.alg.actor, cfg.alg.critic
    checks = {
        "latent OT conditional SAC": actor.distillation_loss == "ot_conditional_sac"
        and actor.ot_student_action == "latent" and not actor.normalize_ot_cost,
        "supported potential optimizer": actor.ot_potential_mode in {"persistent_dual", "fresh_sinkhorn"},
        "canonical persistent dual architecture and rate": actor.ot_potential_mode != "persistent_dual"
        or (list(actor.ot_dual_hidden_dims) == [256, 256] and actor.ot_dual_learning_rate == 1e-4),
        "continuous conditional Gaussian actor": actor.type == "semi_implicit"
        and actor.get("latent_prior", "normal") == "normal",
        "conditional-mixture teacher": actor.teacher_distribution == "conditional_mixture",
        "one teacher per fresh latent": actor.proposal_sampling_mode == "stratified"
        and actor.proposals_per_policy_sample == 1,
        "full fixed density correction": actor.density_correction
        and actor.density_beta == 1.0 and not actor.adaptive_density_beta,
        "scalar SAC soft TD": critic.get("backup_mode") == "soft_td" and critic.n_atoms == 1
        and critic.entr_coeff == 0.0,
        "fixed TD entropy coefficient equals temperature": cfg.alg.ent_coef.type == "const"
        and cfg.alg.ent_coef.init == actor.temperature,
        "16-component marginal-mixture TD entropy estimate": actor.entropy_samples == 16
        and not actor.entropy_diagnostics,
        "no guard or proximal substitute": not actor.soft_guard.enabled
        and actor.get("soft_proximal_ess_fraction", 0.0) == 0.0,
        "both mu-only evaluation modes": cfg.get("dual_mu_eval", False) and cfg.mu_only_eval,
        "no uniform exploration replacement": cfg.alg.behavior_uniform_probability == 0.0,
        "fixed temperature": not actor.get("temperature_schedule", {}).get("enabled", False),
    }
    for name, valid in checks.items():
        if not valid:
            raise ValueError(f"Not the v7 algorithm: {name}")
    return cfg


if __name__ == "__main__":
    cfg = verify(sys.argv[1:], os.environ.get("OPTIQ_CONFIG", "mujoco_v7"))
    actor = cfg.alg.actor
    print(f"PASS: v7 {cfg.env_name}, seed={cfg.seed}, actor={actor.hidden_dims}, "
          f"critic={cfg.alg.critic.hs}, T={actor.temperature}, "
          f"OT={actor.ot_num_latents}x{actor.ot_teacher_resample_count}, "
          f"teacher={actor.teacher_proposal_components * actor.proposals_per_policy_sample}"
          f"->{actor.ot_teacher_resample_count}, actor queries={actor.num_policy_samples}; "
          f"fresh proposal components={actor.teacher_proposal_components}, one per latent; "
          f"potential={actor.ot_potential_mode}; "
          "conditional SAC actor, soft TD with fixed alpha=T, dual mu-only eval")
