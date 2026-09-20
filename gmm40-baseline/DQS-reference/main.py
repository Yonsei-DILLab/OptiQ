import os

from absl import app, flags

from src.utils.logger import Logger
from src.envs import make_dmc_env, make_gmm_env, make_pointmaze_env

from src.trainer import Trainer, GMMTrainer, PointMazeTrainer
from src.agent.dqs import DQS

import yaml
from typing import Dict, Any
from pathlib import Path

os.environ["WANDB_MODE"] = "online"

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', '', 'Experiment description.')
flags.DEFINE_string('env_name', 'GMM', 'Environment name.')
flags.DEFINE_string('wandb_dir', 'dqs_logs', 'Wandb logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('updates_per_step', 1, 'Number of updates per step.')
flags.DEFINE_integer('start_training', int(1e4), 'Number of training steps to start training.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('init_temperature', 1.0, 'Initial temperature for DQS sampling.')
flags.DEFINE_float('final_temperature', 1.0, 'Final temperature for DQS sampling.')
flags.DEFINE_integer('temperature_steps', int(1e6), 'Temperature annealing steps (=final temperature after that).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('debug', False, 'Disable wandb while debugging.')
flags.DEFINE_string('config', None, 'Path to env YAML config to overlay defaults.')
flags.DEFINE_string('wandb_project', None, 'W&B project name (defaults to dqs.<env_name> when set).')
flags.DEFINE_string('wandb_entity', None, 'W&B entity (team or username). If unset, wandb is disabled.')


def _load_yaml(path: Path) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open('r') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top-level: {path}")
    return data


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def main(_):
    # 1) Load defaults, then auto env-specific overrides, then optional user config
    repo_root = Path(__file__).resolve().parent
    defaults_path = repo_root / 'configs' / 'defaults.yaml'
    base_cfg = _load_yaml(defaults_path)
    user_cfg = _load_yaml(Path(FLAGS.config)) if FLAGS.config else {}

    # Decide which env-specific config to overlay (prefer user config, else CLI default)
    selected_env_for_overlay = user_cfg.get('env_name', FLAGS.env_name)

    env_specific_cfg = {}
    if selected_env_for_overlay == 'GMM':
        env_specific_cfg = _load_yaml(repo_root / 'configs' / 'gmm.yaml')
    elif selected_env_for_overlay == 'PointMaze':
        env_specific_cfg = _load_yaml(repo_root / 'configs' / 'pointmaze.yaml')

    cfg = _merge_configs(base_cfg, env_specific_cfg)
    cfg = _merge_configs(cfg, user_cfg)
    # Ensure env_name exists for downstream logic
    cfg.setdefault('env_name', selected_env_for_overlay)

    # 2) Apply CLI overrides when explicitly present
    for name in [
        'exp','env_name','wandb_dir','seed','max_steps','replay_buffer_size',
        'start_training','eval_interval','log_interval','batch_size',
        'init_temperature','final_temperature','temperature_steps',
        'tqdm','debug','updates_per_step','eval_episodes',
        'wandb_project','wandb_entity']:
        try:
            if flags.FLAGS[name].present:
                cfg[name] = getattr(FLAGS, name)
        except KeyError:
            pass

    # 3) Select env factory using cfg
    if cfg['env_name'] == 'GMM':
        env_fn = make_gmm_env('GMM-v0')
    elif cfg['env_name'] == 'PointMaze':
        env_fn = make_pointmaze_env()
    else:
        env_fn = make_dmc_env(env_name=cfg['env_name'])

    # 4) Probe env for dims
    probe_env = env_fn(seed=cfg['seed'])
    try:
        probe_obs = probe_env.reset(seed=cfg['seed'])
    except TypeError:
        probe_obs = probe_env.reset()
    if isinstance(probe_obs, tuple):
        probe_obs = probe_obs[0]
    if isinstance(probe_obs, dict) and 'observation' in probe_obs:
        state_dim = probe_env.observation_space['observation'].shape[0]
    else:
        state_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]

    energy_fn_config = dict(cfg.get('energy_fn', {}))
    energy_fn_config.update({
        'state_dim': state_dim,
        'action_dim': action_dim,
        'seed': cfg['seed'],
    })
    agent = DQS(
        sample_dim=action_dim,
        state_dim=state_dim,
        energy_fn_config=energy_fn_config,
        num_estimator_mc_samples=1000,
        num_integration_steps=cfg.get('dqs_num_integration_steps', 500),
        num_samples_to_sample_from_buffer=cfg['batch_size'],
        buffer_update_period=cfg.get('buffer_update_period', 100),
        use_richardsons=False,
        sigma_min=cfg['sigma_min'],
        sigma_max=cfg['sigma_max'],
        init_temperature=cfg['init_temperature'],
        final_temperature=cfg['final_temperature'],
        temperature_steps=cfg['temperature_steps'],
        diffusion_scale=cfg['diffusion_scale'],
        warm_start_steps=cfg.get('warm_start_steps', 1),
        seed=cfg['seed'],
    )

    # 6) Logger: resolve non-absolute paths relative to repo root
    wandb_dir = Path(cfg['wandb_dir'])
    if not wandb_dir.is_absolute():
        wandb_dir = (repo_root / wandb_dir).resolve()
    wandb_project = cfg.get('wandb_project') or (None if cfg['debug'] else ("dqs." + cfg['env_name']))
    wandb_entity = cfg.get('wandb_entity') or os.getenv('WANDB_ENTITY')
    if (not cfg['debug']) and wandb_project and not wandb_entity:
        print("Warning: W&B entity not provided; W&B logging will be disabled. Set --wandb_entity or WANDB_ENTITY to enable.")
    # If entity is missing or debug true, disable wandb by passing "none"
    project_arg = wandb_project if (wandb_project and wandb_entity and not cfg['debug']) else "none"
    entity_arg = wandb_entity if (wandb_project and wandb_entity and not cfg['debug']) else "none"

    logger = Logger(
        work_dir=str(wandb_dir),
        seed=cfg['seed'],
        project=project_arg,
        entity=entity_arg,
        tags=[],
        group="",
        config=cfg,
        disable_wandb=cfg['debug'],
    )

    # 7) Trainer
    if cfg['env_name'] == 'GMM':
        trainer_fn = GMMTrainer
    elif cfg['env_name'] == 'PointMaze':
        trainer_fn = PointMazeTrainer
    else:
        trainer_fn = Trainer

    trainer = trainer_fn(
        env_fn,
        agent=agent,
        logger=logger,
        max_steps=cfg['max_steps'],
        replay_buffer_size=cfg['replay_buffer_size'],
        start_training=cfg['start_training'],
        eval_interval=cfg['eval_interval'],
        log_interval=cfg['log_interval'],
        batch_size=cfg['batch_size'],
        reward_scale=cfg['reward_scale'],
        num_updates_per_step=cfg['updates_per_step'],
        tqdm_bar=cfg['tqdm'],
        seed=cfg['seed'],
        num_eval_envs=cfg.get('eval_episodes', 10),
    )
    trainer.train()

if __name__ == '__main__':
    app.run(main)
