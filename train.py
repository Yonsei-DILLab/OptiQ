"""Standalone MuJoCo training for iBOLT."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import hydra
from hydra import compose, initialize_config_dir
import jax
from omegaconf import OmegaConf
import wandb
import environment as runner
from ibolt.runtime import provenance

ROOT = Path(__file__).resolve().parent

def compose_config(overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose(config_name='train', overrides=list(overrides))

def validate_config(cfg):
    a = cfg.alg.actor
    if not a.log_std_min <= a.initial_log_std <= a.log_std_max:
        raise ValueError('Initial log scale must be inside its bounds.')
    if a.temperature <= 0 or a.num_policy_samples < 1 or a.proposals_per_policy_sample < 1:
        raise ValueError('Temperature and sample counts must be positive.')
    if not 0 <= a.density_correction_beta <= 1:
        raise ValueError('Density correction beta must be in [0,1].')
    if cfg.total_steps <= max(cfg.alg.learning_starts, a.learning_starts):
        raise ValueError('Training must extend past warmup.')
    if cfg.alg.critic.n_critics != 2:
        raise ValueError('This configuration requires two scalar critics.')
    if a.log_std_min != -5 or a.log_std_max != -1:
        raise ValueError('This release retains the gmm-trg scale bounds [-5,-1].')

def run(cfg):
    validate_config(cfg)
    if cfg.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required; set require_gpu=false for CPU validation.')
    cfg.run_name += '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    cfg.output_root = str((Path(cfg.output_root).expanduser() / cfg.run_name).resolve())
    output = Path(cfg.output_root)
    output.mkdir(parents=True, exist_ok=False)
    config = OmegaConf.to_container(cfg, resolve=True)
    config['runtime'] = provenance()
    config['protocol'] = dict(distribution='box-truncated Gaussian mixture with tanh centers',
        objective='density-corrected value-weighted marginal NLL', ot=False,
        evaluation='zero-z and resampled-z centers, no conditional or external noise')
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    tracking = model = callbacks = None
    success = False
    try:
        if cfg.wandb.activate:
            tracking = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                mode=cfg.wandb.mode, group=cfg.wandb.group, job_type=cfg.wandb.job_type,
                name=cfg.run_name, config=config, dir=str(output), save_code=False)
            tracking.define_metric('env_steps')
            tracking.define_metric('*', step_metric='env_steps')
        model, callbacks = runner.create_algorithm(cfg)
        model.learn(total_timesteps=int(cfg.total_steps), callback=callbacks,
                    progress_bar=bool(cfg.progress_bar), log_interval=int(cfg.log_interval))
        model._save_model()
        if model.logger.name_to_value:
            model.logger.dump(model.num_timesteps)
        result = dict(timesteps=model.num_timesteps, updates=model._n_updates, completed=True)
        (output / 'completed.json').write_text(json.dumps(result, indent=2))
        if tracking is not None:
            tracking.summary.update(result)
        success = True
        print(f'Completed: {output}', flush=True)
    finally:
        if model is not None:
            model.get_env().close()
            model.logger.close()
        if callbacks is not None:
            for cb in callbacks.callbacks:
                if hasattr(cb, 'eval_env'):
                    cb.eval_env.close()
        if tracking is not None:
            wandb.finish(exit_code=0 if success else 1)

@hydra.main(version_base=None, config_path='configs', config_name='train')
def main(cfg):
    if cfg.use_jit:
        run(cfg)
    else:
        with jax.disable_jit():
            run(cfg)

if __name__ == '__main__':
    main()
