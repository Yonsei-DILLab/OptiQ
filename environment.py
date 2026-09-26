"""Single-environment MuJoCo collection and paired mean-action evaluation."""
from pathlib import Path
import gymnasium as gym
import wandb
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from ibolt.dual_evaluation import DualMuEvalCallback
from ibolt.runtime import WandbWriter
from exploration import ExplorationIBOLT

def create_algorithm(cfg):
    env = gym.make(cfg.env_name)
    evaluation = make_vec_env(cfg.env_name,n_envs=1,seed=cfg.seed)
    root = Path(cfg.output_root)
    try:
        model = ExplorationIBOLT('MlpPolicy',env=env,cfg=cfg,
            model_save_path=str(root/'checkpoints'/cfg.run_name),
            save_every_n_steps=int(cfg.checkpoint_interval) or int(cfg.total_steps),
            tensorboard_log=str(root/'tensorboard'/cfg.run_name))
        logger = configure(str(root/'logs'),['stdout','csv','tensorboard'])
        if wandb.run is not None:
            logger.output_formats.append(WandbWriter(wandb.run))
        model.set_logger(logger)
        return model, CallbackList([DualMuEvalCallback(evaluation,cfg,root/'eval'/cfg.run_name)])
    except BaseException:
        env.close()
        evaluation.close()
        raise
