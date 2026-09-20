"""Validate Direct GMM and the inherited v5 contract without training or W&B."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from run_optiq_dime import validate_config
from scripts.verify_v5 import verify as verify_v5


def verify(overrides=(), config_name="mujoco_v5_direct_gmm"):
    if config_name not in {"mujoco_v5_direct_gmm", "v5_direct_gmm/final"}:
        raise ValueError("Use the v5 Direct GMM profile")
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/"configs")):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    validate_config(cfg)
    if cfg.alg.actor.distillation_loss != "direct_gmm_nll":
        raise ValueError("Direct GMM requires direct_gmm_nll")
    # Reuse the established v5 validator without weakening its OT checks.
    reference = verify_v5([*overrides, "alg.actor.distillation_loss=conditional_ot_nll"])
    actual_alg = OmegaConf.to_container(cfg.alg, resolve=True)
    actual_alg["actor"]["distillation_loss"] = "conditional_ot_nll"
    if actual_alg != OmegaConf.to_container(reference.alg, resolve=True):
        raise ValueError("Direct GMM must retain the v5 training contract")
    if not cfg.dual_mu_eval or not cfg.mu_only_eval:
        raise ValueError("Direct GMM retains both mu-only evaluations")
    return cfg


if __name__ == "__main__":
    cfg = verify(sys.argv[1:])
    print(f"PASS: v5 Direct GMM {cfg.env_name}, seed={cfg.seed}, T={cfg.alg.actor.temperature}; "
          "same Gaussian teacher, beta=1 density correction, plain TD, dual mu-only eval; no OT")
