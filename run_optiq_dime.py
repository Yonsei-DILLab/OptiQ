"""Train OptiQ's one-step actor with DIME's critic on DMC or MuJoCo v4."""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
import uuid

import hydra
import jax
import omegaconf
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from wandb.integration.sb3 import WandbCallback

from common.buffers import DMCCompatibleDictReplayBuffer
from models.actor_critic_evaluation_callback import EvalCallback
from optiq_dime import OptiQDIME
from optiq_dime.evaluation import MujocoEvalCallback
from optiq_dime.runtime import ROOT, WandbWriter, load_environment, provenance

DOG_TASKS = {"run", "trot", "walk", "stand"}
MUJOCO_ENVS = {"Ant-v4", "Humanoid-v4"}


def validate_config(cfg):
    is_mujoco = cfg.env_name in MUJOCO_ENVS
    if not is_mujoco and not cfg.env_name.startswith("dm_control/"):
        raise ValueError(f"Unsupported environment: {cfg.env_name}")
    if cfg.env_name.startswith("dm_control/dog-") and cfg.task not in DOG_TASKS:
        raise ValueError(f"Invalid Dog task: {cfg.task}")
    actor = cfg.alg.actor
    if actor.proposal_sampling_mode not in {"stratified", "exact"}:
        raise ValueError("proposal_sampling_mode must be stratified or exact")
    if not 0 <= actor.density_beta <= 1:
        raise ValueError("density_beta must be between 0 and 1")
    if actor.proposal_std <= 0 or actor.proposal_clip <= 0 or actor.temperature <= 0:
        raise ValueError("Proposal scale, clip, and temperature must be positive")
    if actor.num_policy_samples < 1 or actor.proposals_per_policy_sample <= int(actor.include_anchor):
        raise ValueError("At least one random proposal per policy sample is required")
    if cfg.total_steps <= cfg.alg.learning_starts:
        raise ValueError("total_steps must exceed learning_starts to exercise training")
    if cfg.checkpoint_interval < 0 or cfg.num_eval_episodes < 1:
        raise ValueError("Invalid checkpoint/evaluation configuration")
    if cfg.alg.critic.v_min >= cfg.alg.critic.v_max:
        raise ValueError("critic.v_min must be below critic.v_max")
    if is_mujoco:
        if cfg.log_interval < 1:
            raise ValueError("log_interval must be positive")
        if cfg.eval_interval < 1 or not cfg.eval_at_start:
            raise ValueError("The reference protocol evaluates at step 1 and a positive interval")
        if not cfg.wandb.activate or cfg.wandb.mode != "online":
            raise ValueError("MuJoCo experiments require online W&B logging")
        if actor.adaptive_density_beta:
            raise ValueError("This MuJoCo sweep uses fixed beta; disable adaptive_density_beta")
    return is_mujoco


def create_algorithm(cfg: DictConfig):
    import gymnasium as gym

    is_mujoco = cfg.env_name in MUJOCO_ENVS
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
    except BaseException:
        training_env.close()
        eval_env.close()
        raise
    if is_mujoco:
        logger = configure(str(output_root / "logs"), ["stdout", "csv", "tensorboard"])
        if wandb.run is not None:
            logger.output_formats.append(WandbWriter(wandb.run))
        model.set_logger(logger)
        return model, CallbackList([MujocoEvalCallback(eval_env, cfg, eval_dir)])
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
    load_environment()
    cfg = hydra.utils.instantiate(cfg)
    is_mujoco = validate_config(cfg)
    if cfg.get("require_gpu", False) and jax.default_backend() != "gpu":
        raise RuntimeError("JAX GPU backend required; refusing CPU fallback")
    if is_mujoco:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        cfg.run_name = f"{cfg.run_name}_{suffix}"
        cfg.output_root = str((ROOT / cfg.output_root / cfg.run_name).resolve())
        Path(cfg.output_root).mkdir(parents=True, exist_ok=False)
    run = None
    if cfg.wandb.activate:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        if is_mujoco:
            wandb_config["runtime"] = provenance()
        run = wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            name=cfg.run_name,
            config=wandb_config,
            entity=os.environ.get("WANDB_ENTITY") or cfg.wandb.entity,
            mode=cfg.wandb.mode,
            sync_tensorboard=not is_mujoco,
            tags=["optiq", "dime-critic", cfg.env_name, cfg.alg.actor.proposal_sampling_mode],
            dir=cfg.output_root if is_mujoco else None,
            save_code=False,
        )
        if is_mujoco:
            run.define_metric("env_steps")
            run.define_metric("*", step_metric="env_steps")
            (Path(cfg.output_root) / "config.json").write_text(json.dumps(wandb_config, indent=2))
            print(f"W&B: {run.url}", flush=True)
    model = callbacks = None
    try:
        model, callbacks = create_algorithm(cfg)
        if is_mujoco:
            environment_metadata = {
                "id": cfg.env_name,
                "observation_shape": list(model.observation_space.shape),
                "action_shape": list(model.action_space.shape),
                "action_low": model.action_space.low.tolist(),
                "action_high": model.action_space.high.tolist(),
                "max_episode_steps": 1000,
                "replay_actions": "normalized [-1, 1]",
                "random_proposal_count": cfg.alg.actor.num_policy_samples * (
                    cfg.alg.actor.proposals_per_policy_sample - int(cfg.alg.actor.include_anchor)),
                "anchor_count": cfg.alg.actor.num_policy_samples * int(cfg.alg.actor.include_anchor),
            }
            run.config.update({"environment": environment_metadata})
            wandb_config["environment"] = environment_metadata
            (Path(cfg.output_root) / "config.json").write_text(json.dumps(wandb_config, indent=2))
        model.learn(
            total_timesteps=int(cfg.total_steps),
            progress_bar=cfg.get("progress_bar", True),
            callback=callbacks,
            tb_log_name="OptiQDIME",
            log_interval=int(cfg.get("log_interval", 1)),
        )
        if is_mujoco:
            model._save_model()
            if model.logger.name_to_value:
                model.logger.dump(model.num_timesteps)
            evaluation = callbacks.callbacks[0]
            run.summary.update({
                "completed": True, "timesteps": model.num_timesteps,
                "updates": model._n_updates,
                "final_eval_return": float(sum(evaluation.returns[-1]) / len(evaluation.returns[-1])),
                "last_eval_step": evaluation.evaluations_timesteps[-1],
            })
            artifact = wandb.Artifact(f"optiq-{run.id}", type="experiment")
            artifact.add_file(str(Path(cfg.output_root) / "config.json"))
            artifact.add_dir(str(evaluation.directory), name="evaluation")
            for filename in Path(model.model_save_path).glob(f"*_{model.num_timesteps}.msgpack"):
                artifact.add_file(str(filename), name=filename.name)
            run.log_artifact(artifact)
            (Path(cfg.output_root) / "completed.json").write_text(json.dumps({
                "wandb_url": run.url, "timesteps": model.num_timesteps, "updates": model._n_updates,
            }, indent=2))
        print(f"completed env={cfg.env_name} seed={cfg.seed} timesteps={model.num_timesteps} updates={model._n_updates}", flush=True)
    finally:
        if model is not None:
            model.get_env().close()
            model.logger.close()
        if callbacks is not None:
            for callback in callbacks.callbacks:
                if hasattr(callback, "eval_env"):
                    callback.eval_env.close()


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
    load_environment()
    main()
