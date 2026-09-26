from pathlib import Path
import jax
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from wandb.integration.sb3 import WandbCallback
from common.buffers import DMCCompatibleDictReplayBuffer
from models.actor_critic_evaluation_callback import EvalCallback
from ibolt.evaluation import MujocoEvalCallback
from ibolt.dual_evaluation import DualMuEvalCallback
from ibolt.runtime import WandbWriter
from exploration import ExplorationIBOLT as IBOLT
MYOSUITE_ENVS = set()
def is_tracked_environment(cfg):
    return True

def create_algorithm(cfg: DictConfig):
    import gymnasium as gym

    is_mujoco = is_tracked_environment(cfg)
    if cfg.env_name in MYOSUITE_ENVS:
        try:
            import myosuite  # noqa: F401 -- registers the environment with Gymnasium.
        except ImportError as error:
            raise RuntimeError("Install the isolated environment with scripts/setup_no_anchor_env.sh") from error
    training_env = gym.make(cfg.env_name)
    eval_env = make_vec_env(cfg.env_name, n_envs=1, seed=cfg.seed)

    # Preserve DIME's original replay-buffer selection for dog tasks.
    replay_buffer_class = None
    domain = cfg.env_name.split("/", 1)[-1].split("-", 1)[0]
    if domain in {"humanoid", "fish", "walker", "quadruped", "finger"}:
        replay_buffer_class = DMCCompatibleDictReplayBuffer

    output_root = Path(cfg.output_root)
    tensorboard_dir = output_root / "tensorboard" / cfg.run_name
    eval_dir = output_root / "eval" / cfg.run_name
    checkpoint_dir = output_root / "checkpoints" / cfg.run_name
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = None
    save_every_n_steps = 1
    if cfg.checkpoint_interval > 0 or is_mujoco:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = str(checkpoint_dir)
        save_every_n_steps = int(cfg.checkpoint_interval) or int(cfg.total_steps)

    try:
        model = IBOLT(
            "MultiInputPolicy"
            if isinstance(training_env.observation_space, gym.spaces.Dict)
            else "MlpPolicy",
            env=training_env,
            model_save_path=model_save_path,
            save_every_n_steps=save_every_n_steps,
            cfg=cfg,
            tensorboard_log=str(tensorboard_dir),
            replay_buffer_class=replay_buffer_class,
        )
    except BaseException:
        training_env.close()
        eval_env.close()
        raise
    if is_mujoco or cfg.get("dual_mu_eval", False):
        logger = configure(str(output_root / "logs"), ["stdout", "csv", "tensorboard"])
        if wandb.run is not None:
            logger.output_formats.append(WandbWriter(wandb.run))
        model.set_logger(logger)
        callback_class = DualMuEvalCallback if cfg.get("dual_mu_eval", False) else MujocoEvalCallback
        return model, CallbackList([callback_class(eval_env, cfg, eval_dir)])
    eval_callback = EvalCallback(
        eval_env,
        jax_random_key_for_seeds=cfg.seed,
        best_model_save_path=None,
        log_path=str(eval_dir),
        eval_freq=int(cfg.eval_interval),
        n_eval_episodes=int(cfg.num_eval_episodes),
        deterministic=not bool(cfg.stochastic_eval),
        render=False,
    )
    callbacks = [eval_callback]
    if cfg.wandb.activate:
        callbacks.append(WandbCallback(verbose=0))
    return model, CallbackList(callbacks)
