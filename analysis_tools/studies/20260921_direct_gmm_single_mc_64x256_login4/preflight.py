"""No GPU training: resolve all runs and verify existing login4 W&B credentials."""
import importlib.metadata
import time
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from common import ROOT, verify, command, write


def main():
    p, d = verify()
    checked = []
    for env in p['order']:
        for seed in p['seeds']:
            with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'repo/configs')):
                cfg = compose(config_name=p['config_name'], overrides=command(p, env, seed, d['commit'])[3:])
            assert cfg.alg.actor.num_policy_samples == 64
            assert cfg.alg.actor.proposals_per_policy_sample == 4
            assert cfg.alg.actor.temperature == .25
            assert cfg.alg.actor.distillation_loss == 'direct_gmm_nll'
            assert cfg.alg.critic.n_critics == 1 and cfg.alg.critic.backup_samples == 64
            assert cfg.alg.critic.backup_mode == 'td' and cfg.alg.critic.n_atoms == 1
            assert cfg.alg.gamma == .99 and cfg.alg.tau == .005
            assert cfg.alg.batch_size == 256 and cfg.total_steps == 1000000
            assert cfg.eval_interval == 5000 and cfg.num_eval_episodes == 10 and cfg.dual_mu_eval
            assert cfg.wandb.activate and cfg.wandb.mode == 'online'
            assert cfg.wandb.project == p['wandb_project'] and cfg.wandb.group == p['wandb_group']
            write(ROOT / 'resolved_configs' / f'{env}_s{seed}.json', OmegaConf.to_container(cfg, resolve=True))
            checked.append(dict(env=cfg.env_name, seed=seed))
    import wandb
    assert wandb.Api(timeout=30).project(p['wandb_project'], entity=p['wandb_entity']) is not None
    packages = {k: importlib.metadata.version(k) for k in ['jax', 'jaxlib', 'mujoco', 'gymnasium', 'wandb', 'torch']}
    record = dict(passed=True, commit=d['commit'], configs=checked, packages=packages,
                  wandb_project=p['wandb_project'], time=time.time(), training_started=False)
    write(ROOT / 'PREFLIGHT.json', record)
    print('Preflight passed: 5 Humanoid configs, single critic N64/M256/K64/T0.25, existing W&B authentication.')


if __name__ == '__main__':
    main()
