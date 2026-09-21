"""Requested Ant/Humanoid DACER profile; no RL default changes."""
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('priority_dacer_train', HERE.parent/'20260921_gmm_trg_sweep/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import BaseCallback


def compose_config(task, seed, output):
    assert task in ('ant', 'humanoid') and seed in range(4)
    cfg = train.compose_config([f'benchmark={task}', f'seed={seed}',
        'alg.actor.temperature=0.25', 'alg.actor.density_beta=1', 'alg.actor.density_correction_beta=1',
        'dacer.enabled=true', f'dacer.noise_scale={0.15 if task == "humanoid" else 0.1}',
        'wandb.project=gmm-trg', 'wandb.job_type=dacer-exploration',
        f'wandb.group=trg-dacer-priority-20260921-{task}-T0.25-b1',
        f'run_name={task}-trg-dacer-T0.25-b1-s{seed}', f'output_root={output}',
        '+experiment.campaign=trg-dacer-priority-20260921', '+experiment.stage=dacer'])
    reference = train.base.compose_config([f'benchmark={task}'])
    assert OmegaConf.to_container(cfg.alg, resolve=True) == OmegaConf.to_container(reference.alg, resolve=True)
    for key in ('total_steps','eval_interval','num_eval_episodes','dual_mu_eval','mu_only_eval'):
        assert cfg[key] == reference[key]
    return cfg


class Metadata(BaseCallback):
    def _on_step(self): return True
    def _on_training_start(self):
        run = train.runner.wandb.run
        if run is None: return
        env = dict(run.config['environment'])
        env.update(policy='box-truncated N(center=tanh(raw_mu), diagonal sigma), support [-1,1]',
            teacher='uniform conditional box-truncated Gaussian mixture', teacher_hard_cutoff=True,
            teacher_bandwidth_space='normalized action', actor_projection='direct marginal box-truncated Gaussian mixture NLL',
            loss_coordinates='normalized action; differentiable truncation normalizer included',
            entropy_estimator='DACER behavior-only fitted GMM joint entropy proxy',
            exploration='clip(full policy sample + noise_scale * alpha * standard normal, -1, 1)')
        env['evaluation'].update(zero_z='a=center(s,0); no conditional or DACER noise',
            stochastic_z='z~N(0,I) per action; a=center(s,z); no conditional or DACER noise')
        run.config.update({'environment': env}, allow_val_change=True)
        path = Path(run.config['output_root'])/'config.json'
        data = json.loads(path.read_text()); data['environment'] = env
        path.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    task, seed, output = sys.argv[1:]
    cfg = compose_config(task, int(seed), output)
    gpu = int(os.environ['CAMPAIGN_GPU']); cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[gpu::4] or cpus)
    original = train.runner.create_algorithm
    def create(config):
        model, callbacks = original(config)
        callbacks.callbacks.append(Metadata())
        return model, callbacks
    train.runner.create_algorithm = create
    train.compose_config = lambda unused: cfg
    train.main()
