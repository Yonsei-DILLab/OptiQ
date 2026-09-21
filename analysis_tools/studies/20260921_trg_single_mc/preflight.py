"""Resolve exact configs and verify online access before any benchmark starts."""
import sys
import time
from ops import ROOT,ORDER,read,write,verify,environment,command
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf


def main():
    launch=verify()
    checked=[]
    for env in ORDER:
        for seed,m in [(s,m) for s in range(4) for m in (64,256)]:
            with initialize_config_dir(version_base=None,config_dir=str(ROOT/'repo/configs')):
                cfg=compose(config_name='mujoco_trg_single_mc',overrides=command(env,seed,launch['commit'],m)[3:])
            assert cfg.alg.critic.n_critics==1 and cfg.alg.critic.backup_samples==64
            assert cfg.alg.actor.num_policy_samples==64 and cfg.alg.actor.proposals_per_policy_sample==m//64
            assert cfg.alg.actor.temperature==.25 and cfg.alg.actor.distillation_loss=='direct_gmm_nll'
            assert cfg.alg.critic.backup_mode=='td' and cfg.alg.critic.n_atoms==1
            assert cfg.alg.gamma==.99 and cfg.alg.tau==.005 and cfg.alg.batch_size==256
            assert cfg.total_steps==1000000 and cfg.alg.learning_starts==5000
            assert cfg.eval_interval==5000 and cfg.num_eval_episodes==10 and cfg.dual_mu_eval
            assert cfg.alg.optimizer.ac_grad_norm is None
            assert (cfg.alg.actor.log_std_min,cfg.alg.actor.log_std_max,cfg.alg.actor.initial_log_std)==(-5.,-1.,-1.)
            assert cfg.alg.actor.proposal_sampling_mode=='exact'
            write(ROOT/'resolved_configs'/f'{env}_m{m}_s{seed}.json',OmegaConf.to_container(cfg,resolve=True))
            checked.append(dict(env=env,seed=seed,M=m))
    import wandb,os
    os.environ['WANDB_API_KEY']=environment(0)['WANDB_API_KEY']
    assert wandb.Api(timeout=30).project('DirectGMM_heejoon',entity='OptiQ') is not None
    write(ROOT/'PREFLIGHT.json',dict(passed=True,commit=launch['commit'],configs=checked,time=time.time()))
    print('Preflight passed: 24 resolved configs and existing online W&B project',flush=True)


if __name__=='__main__':main()
