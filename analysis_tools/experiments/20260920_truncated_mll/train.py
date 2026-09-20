"""Online 64-component / 64-candidate direct marginal likelihood ablation.

Reuse the existing heejoon Direct GMM implementation, not the legacy top-level
explorer and not OT-conditioned NLL. No archived algorithm files are changed.
"""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / 'analysis_tools/studies/20260918_nonstationary_nd/v5'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))  # dedicated bounded truncated-policy implementation

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import run_optiq_dime as runner

OVERRIDES = [
    'alg.actor.distillation_loss=direct_gmm_nll',
    'alg.actor.num_policy_samples=64', 'alg.actor.proposals_per_policy_sample=1',
    'alg.actor.proposal_sampling_mode=exact', 'alg.actor.temperature=0.25',
    'alg.actor.log_std_max=-1.0', 'alg.actor.initial_log_std=-1.0',
    # This equals exp(log_std_min), so the teacher-only floor is an identity
    # everywhere in the actor's allowed scale range. Keep archived validation.
    'alg.actor.teacher_std_floor=0.006737946999085467',
    'alg.actor.normalize_ot_cost=false', 'alg.actor.entropy_diagnostics=false',
    'alg.optimizer.ac_grad_norm=null', 'alg.behavior_uniform_probability=0.0',
    'progress_bar=false', 'wandb.project=heejoon-truncated-mll',
    'wandb.job_type=direct-marginal-likelihood',
    'run_name=${task}-truncatedMLL-r2-N64-M64-T025-s${seed}',
    'wandb.group=${env_name}_truncatedMLL_r2_N64_M64_T025',
    '+experiment={method:truncated_marginal_nll,distribution:box_truncated_gaussian,center_transform:tanh,bounded_mu:true,ot:false,resampling:false,components:64,candidates:64,teacher_extra_floor:false}',
]


def compose_config(overrides=()):
    with initialize_config_dir(config_dir=str(SOURCE/'configs'), version_base=None):
        cfg = compose(config_name='mujoco_v5', overrides=OVERRIDES + list(overrides))
    runner.validate_config(cfg)
    a = cfg.alg.actor
    assert a.distillation_loss == 'direct_gmm_nll'
    assert a.num_policy_samples == 64 and a.proposals_per_policy_sample == 1
    assert a.temperature == .25 and a.proposal_sampling_mode == 'exact'
    assert a.type == 'semi_implicit' and a.teacher_distribution == 'conditional_mixture'
    assert not a.include_anchor and a.density_beta == 1 and a.density_correction
    assert a.log_std_min == -5. and a.teacher_std_floor == 0.006737946999085467
    assert a.log_std_max == -1. and a.initial_log_std == -1.
    assert cfg.alg.critic.backup_mode == 'td' and cfg.alg.critic.n_atoms == 1
    assert cfg.alg.optimizer.ac_grad_norm is None and cfg.alg.behavior_uniform_probability == 0
    assert list(a.hidden_dims) == list(cfg.alg.critic.hs) == [256, 256]
    return cfg


def main():
    cfg = compose_config(sys.argv[1:])
    archived_provenance = runner.provenance
    def provenance():
        result = archived_provenance()
        # Archived code carries an old source sidecar. Preserve it as ancestry,
        # but identify the actual committed heejoon-based experiment correctly.
        result['archived_source_manifest'] = result.pop('source_manifest', None)
        result.update(
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
            git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()),
            algorithm_source=str((HERE/'optiq_dime').relative_to(REPO)),
        )
        return result
    runner.provenance = provenance
    runner.initialize_and_run(cfg)


if __name__ == '__main__':
    main()
