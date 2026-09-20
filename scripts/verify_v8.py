"""Check the portable v8 RL configuration without starting training or W&B."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from run_optiq_dime import validate_config, v8_algorithm_metadata


def verify(overrides=(), config_name="mujoco_v8"):
    if config_name not in {"mujoco_v8", "v8/final"}:
        raise ValueError("Use mujoco_v8 or v8/final")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    validate_config(cfg)
    a = cfg.alg.actor
    if a.distillation_loss != "ot_gaussian_conditional_sac":
        raise ValueError("v8 requires Gaussian-likelihood OT conditional SAC")
    if a.proposals_per_policy_sample != 1 or a.proposal_sampling_mode != "stratified":
        raise ValueError("v8 requires one teacher action per fresh Gaussian")
    if cfg.alg.critic.backup_mode != "soft_td" or cfg.alg.critic.n_atoms != 1:
        raise ValueError("v8 uses scalar soft TD")
    if a.soft_guard.enabled or a.get("soft_proximal_ess_fraction", 0.) != 0:
        raise ValueError("v8 does not use an alternative guard/proximal objective")
    if cfg.alg.ent_coef.type != "const" or cfg.alg.ent_coef.init != a.temperature:
        raise ValueError("Teacher, actor and soft TD must use the same fixed alpha")
    return cfg


if __name__ == "__main__":
    import json
    cfg = verify(sys.argv[1:])
    print(json.dumps(v8_algorithm_metadata(cfg), indent=2))
