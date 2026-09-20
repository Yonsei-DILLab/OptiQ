"""Compare log-sigma caps 0 and -2 with a matched initial sigma of .1."""
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20260920_truncated_mll'
sys.path.insert(0,str(BASE))
import train as base


def compose_config(cap,overrides=()):
    assert cap in (0.,-2.)
    changes=[f'alg.actor.initial_log_std={math.log(.1)}',
        'alg.actor.log_std_min=-5.0',f'alg.actor.log_std_max={cap}',
        'total_steps=20000','seed=0','eval_interval=1000','diagnostic_interval=1000',
        'wandb.project=abla','wandb.job_type=log-sigma-cap-ablation',
        'wandb.group=${env_name}_trg_logcap_20k_s0_20260921',
        '+experiment.initial_sigma=0.1',f'+experiment.log_sigma_cap={cap}',
        '+experiment.ablation=log_sigma_upper_bound']
    with base.initialize_config_dir(config_dir=str(base.SOURCE/'configs'),version_base=None):
        cfg=base.compose(config_name='mujoco_v5',overrides=base.OVERRIDES+changes+list(overrides))
    base.runner.validate_config(cfg)
    a=cfg.alg.actor
    assert math.isclose(math.exp(a.initial_log_std),.1) and a.log_std_output_init_scale==0
    assert a.log_std_min==-5 and a.log_std_max==cap
    assert a.log_std_min<a.initial_log_std<a.log_std_max
    assert a.get('latent_prior','normal')=='normal'
    assert a.num_policy_samples==64 and a.proposals_per_policy_sample==1
    assert a.distillation_loss=='direct_gmm_nll' and a.temperature==.25
    assert cfg.alg.batch_size==256 and cfg.alg.utd==1 and cfg.alg.policy_delay==1
    assert cfg.alg.optimizer.lr_actor==cfg.alg.optimizer.lr_critic==.0003
    assert cfg.alg.optimizer.ac_grad_norm is None
    assert cfg.alg.learning_starts==a.learning_starts==5000
    assert list(a.hidden_dims)==list(cfg.alg.critic.hs)==[256,256]
    assert cfg.total_steps==20000 and cfg.seed==0
    assert cfg.dual_mu_eval and cfg.num_eval_episodes==10
    return cfg


if __name__=='__main__':
    cap=float(sys.argv[1]);cfg=compose_config(cap,sys.argv[2:])
    base.compose_config=lambda unused:cfg
    try:base.main()
    except BaseException:
        base.runner.wandb.finish(exit_code=1)
        raise
    else:base.runner.wandb.finish(exit_code=0)
