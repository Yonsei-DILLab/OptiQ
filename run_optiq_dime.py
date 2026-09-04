"""Train OptiQ's one-step actor with DIME's critic on DMC Dog tasks."""

from pathlib import Path

import hydra
import jax
import omegaconf
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from common.buffers import DMCCompatibleDictReplayBuffer
from models.actor_critic_evaluation_callback import EvalCallback
from optiq_dime import OptiQDIME

DOG_TASKS = {"run", "trot", "walk", "stand"}


def create_algorithm(cfg: DictConfig):
    import gymnasium as gym

    if cfg.task not in DOG_TASKS:
        raise ValueError(f"task must be one of {sorted(DOG_TASKS)}, got {cfg.task!r}")
    training_env = gym.make(cfg.env_name)
    eval_env = make_vec_env(cfg.env_name, n_envs=1, seed=cfg.seed)

    # Preserve DIME's original replay-buffer selection for dog tasks.
    replay_buffer_class = None
    domain = cfg.env_name.split("/", 1)[1].split("-", 1)[0]
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
    if cfg.checkpoint_interval > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = str(checkpoint_dir)
        save_every_n_steps = int(cfg.checkpoint_interval)

    model = OptiQDIME(
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


def initialize_and_run(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    if cfg.wandb.activate:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            name=cfg.run_name,
            config=wandb_config,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            sync_tensorboard=True,
            tags=[
                "optiq",
                "dime-critic",
                "dmc-dog",
                cfg.alg.actor.proposal_sampling_mode,
                cfg.task,
            ],
        )
    model, callbacks = create_algorithm(cfg)
    model.learn(
        total_timesteps=int(cfg.total_steps),
        progress_bar=True,
        callback=callbacks,
    )
    print(
        f"completed task={cfg.task} seed={cfg.seed} "
        f"timesteps={model.num_timesteps} updates={model._n_updates}",
        flush=True,
    )


@hydra.main(version_base=None, config_path="configs", config_name="optiq_dime_dog")
def main(cfg: DictConfig) -> None:
    try:
        if cfg.use_jit:
            initialize_and_run(cfg)
        else:
            with jax.disable_jit():
                initialize_and_run(cfg)
    except BaseException:
        if cfg.wandb.activate:
            wandb.finish(exit_code=1)
        raise
    else:
        if cfg.wandb.activate:
            wandb.finish(exit_code=0)


if __name__ == "__main__":
    main()
