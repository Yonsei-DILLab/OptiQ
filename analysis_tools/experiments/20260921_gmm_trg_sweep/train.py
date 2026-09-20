"""Frozen direct-gmm-trg campaign; only temperature, beta, behavior regulator vary."""
from pathlib import Path
import importlib.util
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = HERE.parent / '20260920_truncated_mll'
spec = importlib.util.spec_from_file_location('trg_base_train', BASE / 'train.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(HERE))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from regulator import BehaviorRegulatedOptiQ

runner = base.runner
runner.OptiQDIME = BehaviorRegulatedOptiQ
REGULATOR = dict(enabled=False, target_entropy_per_dim=-0.9, initial_alpha=0.27,
                 alpha_lr=0.03, interval_updates=10000, components=3, samples=200,
                 noise_scale=0.1, entropy_seed=42, behavior_only=True)

def compose_config(overrides=()):
    extra = ['wandb.project=gmm-trg', 'wandb.job_type=gmm-trg-sweep',
             'run_name=${task}-gmm-trg-T${alg.actor.temperature}-b${alg.actor.density_beta}-s${seed}',
             'wandb.group=${env_name}_T${alg.actor.temperature}_b${alg.actor.density_beta}',
             '+dacer={' + ','.join(f'{k}:{str(v).lower()}' for k,v in REGULATOR.items()) + '}']
    with initialize_config_dir(config_dir=str(base.SOURCE / 'configs'), version_base=None):
        cfg = compose(config_name='mujoco_v5', overrides=base.OVERRIDES + extra + list(overrides))
    runner.validate_config(cfg)
    a = cfg.alg.actor
    assert a.distillation_loss == 'direct_gmm_nll' and a.num_policy_samples == 64
    assert a.proposals_per_policy_sample == 1 and a.proposal_sampling_mode == 'exact'
    assert a.log_std_min == -5 and a.log_std_max == -1 and a.initial_log_std == -1
    assert a.temperature > 0 and a.density_beta in (0.5, 0.9, 1.0)
    assert not a.include_anchor and a.density_correction and not a.adaptive_density_beta
    assert list(a.hidden_dims) == list(cfg.alg.critic.hs) == [256, 256]
    assert cfg.alg.critic.backup_mode == 'td' and cfg.alg.critic.n_atoms == 1
    assert cfg.alg.optimizer.ac_grad_norm is None and cfg.alg.behavior_uniform_probability == 0
    assert cfg.dacer.behavior_only
    return cfg

def main():
    cfg = compose_config(sys.argv[1:])
    archived = runner.provenance
    def provenance():
        result = archived()
        result['archived_source_manifest'] = result.pop('source_manifest', None)
        result.update(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                      git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
                      git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()),
                      algorithm_source=str((BASE/'optiq_dime').relative_to(REPO)),
                      sweep_source=str(HERE.relative_to(REPO)))
        return result
    runner.provenance = provenance
    try:
        runner.initialize_and_run(cfg)
    finally:
        import wandb
        if wandb.run is not None:
            wandb.finish(exit_code=1 if sys.exc_info()[0] else 0)

if __name__ == '__main__': main()
