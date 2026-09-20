"""Explicit temperature/beta profiles; preserve every other RL default."""
import copy
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '20260920_truncated_mll'
sys.path.insert(0, str(BASE))
import train as base
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import BaseCallback

TEMPERATURES = {'ant': (0.1, 0.05), 'humanoid': (0.5, 0.1)}
BETAS = (1.0, 0.5, 0.9)
ORIGINAL_VALIDATE = base.runner.validate_config


def validate(cfg):
    a = cfg.alg.actor
    if float(a.density_beta) not in BETAS or a.density_correction_beta != a.density_beta:
        raise ValueError('Only the registered fixed beta values are allowed')
    # The historical runner rejects beta != 1 before reaching the already
    # beta-parametric objective. Validate all its other invariants on a copy.
    checked = copy.deepcopy(cfg)
    checked.alg.actor.density_beta = 1.0
    checked.alg.actor.density_correction_beta = 1.0
    return ORIGINAL_VALIDATE(checked)


def compose_config(task, temperature, beta, seed, stage, output_root):
    assert task in TEMPERATURES and temperature in TEMPERATURES[task]
    assert seed in range(5)
    assert (stage == 'temperature' and beta == 1) or (stage == 'beta' and beta in (.5, .9))
    name = f'{task}-trg-{stage}-T{temperature:g}-b{beta:g}-s{seed}'
    overrides = [f'benchmark={task}', f'seed={seed}',
        f'alg.actor.temperature={temperature}', f'alg.actor.density_beta={beta}',
        f'alg.actor.density_correction_beta={beta}', 'wandb.project=gmm-trg',
        f'wandb.job_type={stage}-ablation',
        f'wandb.group=trg-temp-beta-20260921-{task}-{stage}-T{temperature:g}-b{beta:g}',
        f'run_name={name}', f'output_root={output_root}',
        '+experiment.campaign=trg-temp-beta-20260921', f'+experiment.stage={stage}',
        '+experiment.selection_metric=stochastic_z_last_100k_mean',
        '+experiment.selection_window_steps=100000',
        '+experiment.latent_prior=normal', '+experiment.density_beta_ablation=true']
    with base.initialize_config_dir(config_dir=str(base.SOURCE/'configs'), version_base=None):
        cfg = base.compose(config_name='mujoco_v5', overrides=base.OVERRIDES+overrides)
    validate(cfg)
    # Check the entire algorithm configuration against the current default,
    # permitting only the two user-requested independent variables.
    reference = base.compose_config([f'benchmark={task}'])
    actual = OmegaConf.to_container(cfg.alg, resolve=True)
    expected = OmegaConf.to_container(reference.alg, resolve=True)
    for key in ('temperature', 'density_beta', 'density_correction_beta'):
        actual['actor'].pop(key); expected['actor'].pop(key)
    assert actual == expected, 'An unrequested algorithm default changed'
    assert cfg.total_steps == reference.total_steps == 1_000_000
    for key in ('eval_interval', 'num_eval_episodes', 'eval_at_start', 'diagnostic_interval',
                'checkpoint_interval', 'dual_mu_eval', 'mu_only_eval', 'use_jit', 'require_gpu'):
        assert cfg[key] == reference[key], key
    return cfg


class CorrectTruncatedMetadata(BaseCallback):
    """Correct legacy description strings, without changing training/evaluation."""
    def _on_training_start(self):
        run = base.runner.wandb.run
        if run is None:
            return
        env = dict(run.config['environment'])
        env.update(policy='box-truncated N(center=tanh(raw_mu), diagonal sigma), support [-1,1]',
            teacher='uniform conditional box-truncated Gaussian mixture',
            teacher_bandwidth_space='normalized action', teacher_hard_cutoff=True,
            actor_projection='direct marginal box-truncated Gaussian mixture NLL',
            loss_coordinates='normalized action; differentiable truncation normalizer included')
        env['evaluation'].update(zero_z='a=center(s,0); no conditional noise',
            stochastic_z='z~N(0,I) per action; a=center(s,z); no conditional noise')
        run.config.update({'environment': env}, allow_val_change=True)
        path = Path(run.config['output_root'])/'config.json'
        data = json.loads(path.read_text()); data['environment'] = env
        path.write_text(json.dumps(data, indent=2)+'\n')

    def _on_step(self):
        return True


def main():
    task, temp, beta, seed, stage, output = sys.argv[1:]
    cfg = compose_config(task, float(temp), float(beta), int(seed), stage, output)
    gpu = int(os.environ['CAMPAIGN_GPU'])
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[gpu::4] or cpus)
    original_create = base.runner.create_algorithm
    def create(config):
        model, callbacks = original_create(config)
        callbacks.callbacks.append(CorrectTruncatedMetadata())
        return model, callbacks
    base.runner.create_algorithm = create
    base.runner.validate_config = validate
    base.compose_config = lambda unused: cfg
    try:
        base.main()
        from campaign import summarize_run
        run = base.runner.wandb.run
        metrics = summarize_run(Path(run.config['output_root']))
        run.summary.update({f'selection/{k}': v for k,v in metrics.items() if isinstance(v,(int,float))})
    except BaseException:
        base.runner.wandb.finish(exit_code=1)
        raise
    else:
        base.runner.wandb.finish(exit_code=0)


if __name__ == '__main__':
    main()
