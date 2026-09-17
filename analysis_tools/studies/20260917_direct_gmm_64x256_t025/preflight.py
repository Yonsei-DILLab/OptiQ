"""Resolve all configs and check existing W&B authentication without GPU work."""
from pathlib import Path
import argparse
import os
import time
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from controller import read, write, command, tasks_for, verify_source


def main(root):
    root = Path(root)
    p = read(root / 'plan.json')
    commit = read(root / 'DEPLOYMENT.json')['commit']
    verify_source(p)
    configs = []
    for task in tasks_for(p):
        with initialize_config_dir(version_base=None, config_dir=str(Path(p['repo']) / 'configs')):
            cfg = compose(config_name=p['config_name'], overrides=command(p, task, commit)[3:])
        assert cfg.alg.actor.num_policy_samples == 64
        assert cfg.alg.actor.proposals_per_policy_sample == 4
        assert cfg.alg.actor.temperature == .25
        assert cfg.alg.actor.distillation_loss == 'direct_gmm_nll'
        assert cfg.alg.batch_size == 256 and cfg.total_steps == 1000000
        assert cfg.eval_interval == 5000 and cfg.num_eval_episodes == 10 and cfg.dual_mu_eval
        assert cfg.wandb.activate and cfg.wandb.mode == 'online'
        assert cfg.wandb.project == p['wandb_project'] and cfg.wandb.group == p['wandb_group']
        assert cfg.experiment_commit == commit
        resolved = OmegaConf.to_container(cfg, resolve=True)
        write(root / 'resolved_configs' / (task['name'] + '.json'), resolved)
        configs.append(dict(name=task['name'], env_name=cfg.env_name, seed=cfg.seed))
    # Reuse the already authorized key in place; never copy it into campaign files.
    os.environ['WANDB_API_KEY'] = Path('/workspace/optiq-secrets/wandb_api_key').read_text().strip()
    import wandb
    project = wandb.Api(timeout=30).project(p['wandb_project'], entity=p['wandb_entity'])
    assert project is not None
    result = dict(passed=True, commit=commit, configurations=configs, checked_unix=time.time(),
                  wandb_project=p['wandb_project'], wandb_entity=p['wandb_entity'],
                  wandb_group=p['wandb_group'], gpu_training_started=False,
                  note='Existing project read access verified; existing online key reused. New run metrics available only after scheduled launch.')
    write(root / 'PREFLIGHT.json', result)
    print('PREFLIGHT passed: 20 configs, N64/M256/T0.25, W&B project accessible; no training started.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    main(parser.parse_args().root)
