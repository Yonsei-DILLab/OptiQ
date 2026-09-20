"""v7 routes a new actor objective while preserving explicit historical profiles."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from run_optiq_dime import validate_config, v7_algorithm_metadata
from scripts.verify_v5 import verify as verify_v5
from scripts.verify_v7 import verify

ROOT = Path(__file__).resolve().parents[1]


def config(overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="mujoco_v7", overrides=list(overrides))


def test_v7_changes_actor_and_soft_td_contract_but_preserves_v5_training_budget():
    actual = OmegaConf.to_container(verify(["benchmark=ant"]), resolve=True)
    alias = OmegaConf.to_container(verify(["benchmark=ant"], "v7/final"), resolve=True)
    expected = OmegaConf.to_container(verify_v5(["benchmark=ant"]), resolve=True)
    expected["alg"]["actor"].update(
        distillation_loss="ot_conditional_sac", ot_student_action="latent",
        latent_prior="normal", normalize_ot_cost=False,
        ot_num_latents=4096, ot_teacher_resample_count=16, ot_latent_seed=0,
        proposals_per_policy_sample=1, teacher_proposal_components=256,
        proposal_sampling_mode="stratified", entropy_samples=16,
        ot_potential_mode="persistent_dual", ot_dual_hidden_dims=[256, 256],
        ot_dual_learning_rate=1e-4,
    )
    expected["alg"]["critic"]["backup_mode"] = "soft_td"
    expected["alg"]["ent_coef"]["init"] = .25
    for key in ("run_name", "output_root", "wandb"):
        expected[key] = actual[key]
    assert actual == alias == expected
    assert actual["wandb"]["project"] == "v7"
    actor, optimizer = actual["alg"]["actor"], actual["alg"]["optimizer"]
    assert actor["num_policy_samples"] == 16 and actor["proposals_per_policy_sample"] == 1
    assert actor["teacher_proposal_components"] == 256
    assert actor["teacher_proposal_components"] * actor["proposals_per_policy_sample"] == 256
    assert actor["proposal_sampling_mode"] == "stratified"
    assert actor["ot_num_latents"] == 4096 and actor["ot_teacher_resample_count"] == 16
    assert actor["ot_latent_seed"] == actual["seed"] == 0
    assert actor["ot_potential_mode"] == "persistent_dual"
    assert actor["ot_dual_hidden_dims"] == [256, 256] and actor["ot_dual_learning_rate"] == 1e-4
    assert actor["sinkhorn_epsilon"] == 0.1 and actor["sinkhorn_iterations"] == 100
    assert actor["temperature"] == 0.25 and actor["teacher_std_floor"] == 0.05
    assert optimizer["lr_actor"] == optimizer["lr_critic"] == 0.0003
    assert actual["alg"]["utd"] == actual["alg"]["policy_delay"] == 1


@pytest.mark.parametrize("override", [
    "alg.actor.ot_student_action=mean", "alg.actor.ot_student_action=sample",
    "alg.actor.distillation_loss=conditional_ot_nll",
    "alg.actor.distillation_loss=ot_marginal_sac",
    "alg.actor.normalize_ot_cost=true", "alg.actor.teacher_distribution=realized_kde",
    "alg.actor.type=implicit", "alg.actor.latent_prior=finite",
    "alg.actor.density_correction=false", "alg.actor.density_correction_beta=0.5",
    "alg.actor.adaptive_density_beta=true", "alg.critic.backup_mode=td",
    "alg.ent_coef.init=0.0", "alg.ent_coef.type=auto", "alg.actor.entropy_samples=0",
    "alg.actor.ot_num_latents=0", "alg.actor.ot_num_latents=1.5",
    "alg.actor.ot_teacher_resample_count=0", "alg.actor.ot_teacher_resample_count=-1",
    "alg.actor.ot_teacher_resample_count=2.5", "alg.actor.ot_latent_seed=-1",
    "alg.actor.ot_teacher_resample_count=32", "alg.actor.num_policy_samples=32",
    "alg.actor.ot_latent_seed=4294967296", "alg.actor.ot_latent_seed=1.5",
    "alg.actor.teacher_proposal_components=0", "alg.actor.teacher_proposal_components=1.5",
    "alg.actor.ot_potential_mode=unknown", "alg.actor.ot_dual_learning_rate=0",
    "alg.actor.ot_dual_learning_rate=-0.1", "alg.actor.ot_dual_learning_rate=.nan",
    "alg.actor.ot_dual_hidden_dims=[]", "alg.actor.ot_dual_hidden_dims=[256,0]",
    "alg.actor.ot_dual_hidden_dims=[32,1.5]", "alg.actor.ot_dual_hidden_dims=256",
])
def test_validation_rejects_incompatible_actor_or_backup(override):
    with pytest.raises(ValueError):
        validate_config(config([override]))


@pytest.mark.parametrize("override", [
    "alg.actor.soft_guard.enabled=true", "alg.actor.entropy_samples=8",
    "alg.actor.entropy_diagnostics=true", "dual_mu_eval=false",
    "alg.behavior_uniform_probability=0.1",
    "alg.actor.proposal_sampling_mode=exact", "alg.actor.proposals_per_policy_sample=2",
    "alg.actor.ot_dual_hidden_dims=[128,128]", "alg.actor.ot_dual_learning_rate=0.001",
    "+alg.actor.temperature_schedule={enabled:true,final_temperature:0.25,anneal_steps:40000}",
])
def test_v7_launcher_rejects_other_profile_variants(override):
    with pytest.raises(ValueError):
        verify([override])


@pytest.mark.parametrize("temperature", [0.01, 0.1, 0.25, 1.0])
def test_temperature_override_updates_actor_and_soft_td_together_without_auto_alpha(temperature):
    cfg = verify([f"alg.actor.temperature={temperature}", "benchmark=hopper", "seed=4"])
    assert cfg.alg.actor.temperature == temperature
    assert cfg.alg.ent_coef.type == "const" and cfg.alg.ent_coef.init == temperature
    assert cfg.alg.critic.entr_coeff == 0.0  # Categorical entropy remains disabled.
    assert cfg.alg.critic.backup_mode == "soft_td"
    assert cfg.alg.actor.entropy_samples == 16
    assert cfg.alg.actor.ot_latent_seed == cfg.seed == 4


def test_explicit_v5_remains_full_row_mean_action_nll():
    cfg = verify_v5(["benchmark=ant"])
    assert cfg.alg.actor.ot_student_action == "mean"
    assert cfg.alg.actor.distillation_loss == "conditional_ot_nll"
    assert cfg.alg.critic.backup_mode == "td" and cfg.alg.ent_coef.init == 0.0
    with pytest.raises(ValueError, match="v7 launcher"):
        verify(config_name="mujoco_v5")


def test_metadata_distinguishes_conditional_actor_entropy_from_marginal_soft_td():
    cfg = verify(["alg.actor.temperature=0.1"])
    metadata = v7_algorithm_metadata(cfg)
    assert metadata["actor_entropy_coefficient"] == 0.1
    assert metadata["assignment_temperature"] == 0.1
    assert metadata["backup_entropy_coefficient"] == 0.1
    assert not metadata["actor_entropy_auto_tuning"]
    assert not metadata["backup_entropy_auto_tuning"]
    assert metadata["critic_backup"] == "soft_td"
    assert metadata["backup_entropy_components"] == 16
    assert "conditional" in metadata["actor_entropy_estimator"]
    assert "marginal" in metadata["backup_entropy_estimator"]
    assert "estimate" in metadata["backup_entropy_estimator"]
    assert metadata["teacher_resampled_before_ot"]
    assert metadata["ot_source_count"] == 4096
    assert metadata["teacher_candidate_count"] == 256
    assert metadata["teacher_proposal_component_count"] == 256
    assert metadata["teacher_one_per_latent"]
    assert metadata["teacher_sampling_mode"] == "stratified"
    assert metadata["ot_teacher_count"] == metadata["actor_query_count"] == 16
    assert "uniform" in metadata["ot_teacher_marginal"]
    assert "no second importance weight" in metadata["ot_teacher_marginal"]
    assert "one i sampled from Pr(i|b_tilde_j,s)" in metadata["actor_source_selection"]
    assert "no clipping or self-normalization" in metadata["actor_source_prior_correction"]
    assert metadata["ot_latent_seed"] == cfg.seed
    assert metadata["ot_potential_mode"] == "persistent_dual"
    assert metadata["ot_dual_state_persistent"]
    assert metadata["ot_dual_hidden_dims"] == [256, 256]
    assert metadata["ot_dual_learning_rate"] == 1e-4
    assert "pre-update" in metadata["ot_dual_update_order"]
    assert metadata["actor_q_aggregation"] == metadata["teacher_q_aggregation"] == "mean"
    assert "latent z" in metadata["ot_cost"]
    assert "no NLL" in metadata["actor_projection"]
    assert "- T * log Pr(i|a,s)" in metadata["actor_objective"]
    assert v7_algorithm_metadata(verify_v5()) == {}


def test_tiny_integration_bank_can_be_used_for_explicit_validation():
    cfg = verify(["alg.actor.ot_num_latents=32", "alg.actor.ot_latent_seed=17"])
    assert cfg.alg.actor.ot_num_latents == 32
    assert cfg.alg.actor.entropy_samples == 16
    assert cfg.alg.actor.ot_latent_seed == 17


def test_fresh_sinkhorn_remains_an_explicit_comparison_without_persistent_state():
    cfg = verify(["alg.actor.ot_potential_mode=fresh_sinkhorn"])
    metadata = v7_algorithm_metadata(cfg)
    assert metadata["ot_potential_mode"] == "fresh_sinkhorn"
    assert not metadata["ot_dual_state_persistent"]
    assert metadata["ot_recomputed"] == "fresh Sinkhorn each actor update"
