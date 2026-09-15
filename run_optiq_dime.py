"""Train OptiQ; default: v4 256x2 networks with dual mu-only evaluation."""

import json
import math
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
from optiq_dime.behavior import parse_behavior_best_of_k
from optiq_dime.evaluation import MujocoEvalCallback
from optiq_dime.dual_evaluation import DualMuEvalCallback
from optiq_dime.temperature import parse_temperature_schedule
from optiq_dime.runtime import ROOT, WandbWriter, load_environment, provenance

DOG_TASKS = {"run", "trot", "walk", "stand"}
MUJOCO_ENVS = {"Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4", "Humanoid-v4"}
MYOSUITE_ENVS = {
    "myoHandPenTwirlRandom-v0",
    "myoHandReachRandom-v0",
    "myoHandObjHoldRandom-v0",
}


def is_tracked_environment(cfg):
    return cfg.env_name in MUJOCO_ENVS | MYOSUITE_ENVS


def validate_config(cfg):
    is_mujoco = is_tracked_environment(cfg)
    parse_behavior_best_of_k(cfg.alg)
    if not 0.0 <= float(cfg.alg.get("behavior_uniform_probability", 0.0)) <= 1.0:
        raise ValueError("behavior_uniform_probability must be between 0 and 1")
    if not is_mujoco and not cfg.env_name.startswith("dm_control/"):
        raise ValueError(f"Unsupported environment: {cfg.env_name}")
    if cfg.env_name.startswith("dm_control/dog-") and cfg.task not in DOG_TASKS:
        raise ValueError(f"Invalid Dog task: {cfg.task}")
    actor = cfg.alg.actor
    winner_teacher = actor.get("teacher_distribution") == "best_of_k_winners"
    proposal_k = actor.get("proposal_best_of_k", 1)
    guided_fraction = actor.get("proposal_guided_fraction", 0.0)
    if isinstance(proposal_k, bool) or not isinstance(proposal_k, int) or proposal_k < 1:
        raise ValueError("proposal_best_of_k must be a positive integer")
    if not math.isfinite(float(guided_fraction)) or not 0 <= guided_fraction < 1:
        raise ValueError("proposal_guided_fraction must be finite in [0,1)")
    if (proposal_k == 1) != (guided_fraction == 0):
        raise ValueError("Enable best-k proposals with k>1 and 0<guided_fraction<1")
    if proposal_k > 1 and (
        actor.get("teacher_distribution") != "conditional_mixture"
        or actor.get("type") != "semi_implicit" or not actor.density_correction
        or actor.density_beta != 1.0 or actor.adaptive_density_beta or actor.include_anchor
        or actor.proposal_sampling_mode != "exact" or cfg.alg.critic.n_atoms != 1
        or cfg.alg.critic.n_critics != 2 or actor.get("soft_proximal_ess_fraction", 0.) != 0):
        raise ValueError("Best-k proposals preserve conditional-mixture Boltzmann teacher with full beta=1 correction")
    if cfg.get("dual_mu_eval", False):
        if actor.get("type") != "semi_implicit" or actor.get("latent_prior", "normal") != "normal":
            raise ValueError("dual_mu_eval requires a continuous-latent semi-implicit actor")
    guard = actor.get("soft_guard", {})
    backup_mode = cfg.alg.critic.get(
        "backup_mode", "soft_td" if actor.get("type") == "semi_implicit" else "td"
    )
    if backup_mode not in {"td", "soft_td"}:
        raise ValueError("critic.backup_mode must be td or soft_td")
    if winner_teacher:
        k = actor.get("teacher_best_of_k")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k != cfg.alg.get("behavior_best_of_k", 1):
            raise ValueError("teacher_best_of_k must be a positive integer matching behavior_best_of_k")
        if (actor.temperature is not None or actor.get("temperature_schedule", {}).get("enabled", False)
            or actor.density_correction or actor.density_beta != 0 or actor.adaptive_density_beta):
            raise ValueError("Winner teacher requires temperature=null and no density correction or annealing")
        if (actor.get("latent_prior", "normal") != "normal" or actor.get("type") != "semi_implicit"
            or actor.get("ot_student_action") != "mean" or actor.get("distillation_loss") != "conditional_ot_nll"
            or actor.normalize_ot_cost or actor.source_q_eval != "min" or backup_mode != "td"
            or cfg.alg.critic.n_atoms != 1 or actor.get("source_reference") != "winner_distribution"
            or actor.get("teacher_std_floor") != 0 or actor.proposal_std != 0
            or cfg.alg.get("behavior_uniform_probability", 0.) != 0.):
            raise ValueError("Winner teacher requires continuous mean OT/full NLL, live scalar twin-min, plain TD, no sigma floor or uniform replacement")
    parse_temperature_schedule(actor, backup_mode)
    if backup_mode == "soft_td" and actor.get("type") != "semi_implicit":
        raise ValueError("Soft TD requires a conditional Gaussian policy")
    if backup_mode == "td":
        if guard.get("enabled", False):
            raise ValueError("Plain TD must not use the soft-value acceptance guard")
        if cfg.alg.ent_coef.type != "const" or cfg.alg.ent_coef.init != 0.0:
            raise ValueError("Plain TD requires a constant zero policy entropy coefficient")
    proximal = float(actor.get("soft_proximal_ess_fraction", 0.0))
    if not math.isfinite(proximal) or not 0 <= proximal <= 1:
        raise ValueError("soft_proximal_ess_fraction must be finite and between 0 and 1")
    if proximal > 0:
        if actor.get("latent_prior") != "finite" or actor.get("teacher_distribution") != "conditional_mixture":
            raise ValueError("Proximal extraction requires the actual finite conditional mixture")
        if proximal*actor.num_policy_samples*actor.proposals_per_policy_sample < 1:
            raise ValueError("Proximal minimum ESS must be at least one candidate")
    if actor.get("latent_prior", "normal") not in {"normal", "finite"}:
        raise ValueError("latent_prior must be normal or finite")
    if actor.get("latent_prior", "normal") == "finite":
        count = float(actor.get("latent_components", 0))
        if not math.isfinite(count) or count < 2 or int(count) != count:
            raise ValueError("latent_components must be an integer >=2")
        if actor.get("type") != "semi_implicit":
            raise ValueError("Finite latent policy requires a conditional Gaussian actor")
        if actor.num_policy_samples != count or (backup_mode == "soft_td" and actor.entropy_samples != count):
            raise ValueError("Finite policy requires all components in OT and entropy density")
        if guard.get("enabled", False) and guard.get("components", 16) != count:
            raise ValueError("Finite policy guard requires the entire actual mixture")
        code_seed = float(actor.get("latent_codebook_seed", -1))
        if not math.isfinite(code_seed) or int(code_seed) != code_seed or not 0 <= code_seed < 2**32:
            raise ValueError("latent_codebook_seed must be a uint32 integer")
    skip_scale = float(actor.get("mean_latent_skip_scale", 0.0))
    if not math.isfinite(skip_scale) or skip_scale < 0:
        raise ValueError("mean_latent_skip_scale must be finite and nonnegative")
    if guard.get("enabled", False):
        if actor.get("type") != "semi_implicit":
            raise ValueError("Sampled soft guard requires a semi-implicit actor")
        for name, minimum, default in (("batch_size", 2, 32), ("components", 1, 16), ("draws", 2, 8)):
            value = float(guard.get(name, default))
            if not math.isfinite(value) or value < minimum or int(value) != value:
                raise ValueError(f"soft_guard.{name} must be an integer >= {minimum}")
        multiplier = float(guard.get("standard_error_multiplier", 2.0))
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("soft_guard.standard_error_multiplier must be finite and nonnegative")
    if actor.get("teacher_distribution", "realized_kde") not in {"realized_kde", "conditional_mixture", "best_of_k_winners"}:
        raise ValueError("Unknown teacher distribution")
    if actor.get("teacher_distribution") == "conditional_mixture" and actor.get("type") != "semi_implicit":
        raise ValueError("Conditional mixture teacher requires a semi-implicit actor")
    ot_student_action = actor.get("ot_student_action", "sample")
    if ot_student_action not in {"sample", "mean"}:
        raise ValueError("ot_student_action must be sample or mean")
    if ot_student_action == "mean" and (
        actor.get("type") != "semi_implicit"
        or actor.get("teacher_distribution") not in {"conditional_mixture", "best_of_k_winners"}
        or actor.get("distillation_loss") != "conditional_ot_nll"
    ):
        raise ValueError("Mean-action OT requires a conditional-mixture teacher and conditional OT NLL")
    if actor.get("type", "implicit") not in {"implicit", "semi_implicit"}:
        raise ValueError("actor.type must be implicit or semi_implicit")
    if actor.get("type", "implicit") == "semi_implicit":
        if actor.include_anchor or (not winner_teacher and (not actor.density_correction or actor.density_beta != 1.0)):
            raise ValueError("v2 requires no teacher anchors and full beta=1 density correction")
        if actor.adaptive_density_beta:
            raise ValueError("v2 requires fixed density beta")
        if backup_mode == "soft_td" and (actor.entropy_samples < 1 or int(actor.entropy_samples) != actor.entropy_samples):
            raise ValueError("entropy_samples must be a positive integer")
        if backup_mode == "td" and (actor.entropy_samples < 0 or int(actor.entropy_samples) != actor.entropy_samples):
            raise ValueError("entropy_samples must be a nonnegative integer (unused by plain TD)")
        if not all(math.isfinite(float(actor[k])) for k in
                   ("log_std_min", "log_std_max", "initial_log_std", "proposal_std_pretanh") + (() if winner_teacher else ("temperature",))):
            raise ValueError("v2 scales and log-std limits must be finite")
        if not actor.log_std_min <= actor.initial_log_std <= actor.log_std_max or actor.log_std_min >= actor.log_std_max:
            raise ValueError("initial_log_std must lie within ordered log_std limits")
        if any(not math.isfinite(float(actor[k])) or actor[k] < 0 for k in
               ("mean_output_init_scale", "log_std_output_init_scale")):
            raise ValueError("Output initialization scales must be finite and nonnegative")
        if (not winner_teacher and actor.proposal_std_pretanh <= 0) or actor.proposal_std != actor.proposal_std_pretanh:
            raise ValueError("proposal_std must equal the positive pre-tanh KDE bandwidth")
        if actor.td_noise_std != 0 or actor.td_noise_clip != 0:
            raise ValueError("Conditional Gaussian backups must not add TD smoothing")
        if backup_mode == "soft_td" and (cfg.alg.ent_coef.type != "const" or cfg.alg.ent_coef.init != actor.temperature):
            raise ValueError("v2 backup entropy coefficient must equal the Boltzmann temperature")
        if cfg.alg.critic.get("crossq_style", True):
            raise ValueError("v2 uses target critics; crossq_style must be false")
    if actor.sinkhorn_iterations < 1 or not math.isfinite(float(actor.sinkhorn_epsilon)) or actor.sinkhorn_epsilon <= 0:
        raise ValueError("Sinkhorn epsilon and iteration count must be positive")
    if actor.get("distillation_loss", "pointwise_mse") not in {"pointwise_mse", "conditional_ot_nll"}:
        raise ValueError("distillation_loss must be pointwise_mse or conditional_ot_nll")
    if actor.get("distillation_loss") == "conditional_ot_nll" and actor.get("type") != "semi_implicit":
        raise ValueError("Conditional OT NLL requires a semi-implicit actor")
    if "density_correction_beta" in actor and actor.density_correction_beta != actor.density_beta:
        raise ValueError("density_correction_beta and density_beta must agree")
    if actor.get("learning_starts", cfg.alg.learning_starts) < cfg.alg.learning_starts:
        raise ValueError("Actor learning_starts must be at least alg.learning_starts")
    if actor.get("learning_starts", cfg.alg.learning_starts) >= cfg.total_steps:
        raise ValueError("total_steps must exceed actor.learning_starts")
    if cfg.get("diagnostic_interval", 0) < 0:
        raise ValueError("diagnostic_interval cannot be negative")
    if cfg.env_name in MYOSUITE_ENVS and cfg.get("successful_steps", 0) < 1:
        raise ValueError("MyoSuite requires positive successful_steps")
    if actor.proposal_sampling_mode not in {"stratified", "exact"}:
        raise ValueError("proposal_sampling_mode must be stratified or exact")
    if not 0 <= actor.density_beta <= 1:
        raise ValueError("density_beta must be between 0 and 1")
    if actor.proposal_clip <= 0 or (not winner_teacher and (actor.proposal_std <= 0 or actor.temperature <= 0)):
        raise ValueError("Proposal scale, clip, and temperature must be positive")
    if actor.num_policy_samples < 1 or actor.proposals_per_policy_sample <= int(actor.include_anchor):
        raise ValueError("At least one random proposal per policy sample is required")
    if cfg.total_steps <= cfg.alg.learning_starts:
        raise ValueError("total_steps must exceed learning_starts to exercise training")
    if cfg.checkpoint_interval < 0 or cfg.num_eval_episodes < 1:
        raise ValueError("Invalid checkpoint/evaluation configuration")
    critic = cfg.alg.critic
    if critic.n_atoms < 1:
        raise ValueError("critic.n_atoms must be positive")
    critic_type = critic.get("type", "scalar" if critic.n_atoms == 1 else "categorical")
    if critic_type not in {"scalar", "categorical"}:
        raise ValueError("critic.type must be scalar or categorical")
    if (critic_type == "scalar") != (critic.n_atoms == 1):
        raise ValueError("Scalar critics require n_atoms=1; categorical critics require n_atoms>1")
    if critic.n_atoms == 1 and critic.entr_coeff != 0:
        raise ValueError("Scalar critics require entr_coeff=0 (no categorical entropy)")
    if critic.n_atoms > 1 and critic.v_min >= critic.v_max:
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
            tags=["optiq", "scalar-critic" if cfg.alg.critic.n_atoms == 1 else "dime-critic",
                  cfg.env_name, cfg.alg.actor.proposal_sampling_mode,
                  cfg.alg.actor.get("type", "implicit")],
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
                "max_episode_steps": model.get_env().get_attr("spec")[0].max_episode_steps,
                "replay_actions": "normalized [-1, 1]",
                "random_proposal_count": cfg.alg.actor.num_policy_samples * (
                    cfg.alg.actor.proposals_per_policy_sample - int(cfg.alg.actor.include_anchor)),
                "anchor_count": cfg.alg.actor.num_policy_samples * int(cfg.alg.actor.include_anchor),
            }
            if cfg.env_name in MYOSUITE_ENVS:
                environment_metadata["success_criterion"] = f"sum(solved) > {cfg.successful_steps} per episode"
            if cfg.alg.actor.get("type", "implicit") == "semi_implicit":
                soft_backup = model.backup_mode == "soft_td"
                environment_metadata.update(
                    policy="tanh(mu(s,z)+sigma(s,z)*eps)",
                    critic_backup=model.backup_mode,
                    backup_entropy_coefficient=float(cfg.alg.actor.temperature) if soft_backup else 0.0,
                    teacher_temperature=cfg.alg.actor.temperature,
                    entropy_estimator=("IDAC self-inclusive conditional mixture in normalized action coordinates"
                                       if soft_backup else "disabled; action sampling only"),
                    entropy_components=int(cfg.alg.actor.entropy_samples) if soft_backup else 0,
                    policy_entropy_diagnostics=bool(cfg.alg.actor.get("entropy_diagnostics", True)),
                    teacher=("conditional Gaussian mixture with teacher-only minimum std" if
                        cfg.alg.actor.get("teacher_distribution") == "conditional_mixture" else
                        "separate Gaussian KDE centered on realized student pre-tanh samples"),
                    teacher_bandwidth_space="pre-tanh", teacher_hard_cutoff=False,
                    backup_policy="current actor", td_smoothing=False,
                    ot_cost="mean-normalized squared action distance" if cfg.alg.actor.normalize_ot_cost else "squared action distance",
                    actor_projection=("full OT conditional Gaussian likelihood" if
                        cfg.alg.actor.distillation_loss == "conditional_ot_nll" else
                        f"{cfg.alg.actor.transport_target_mode} pointwise MSE"),
                    update_acceptance=("sampled soft-value filter; not a global certificate" if
                        cfg.alg.actor.get("soft_guard", {}).get("enabled", False) else "unfiltered"),
                )
                if soft_backup and cfg.alg.actor.get("latent_prior") == "finite":
                    environment_metadata["entropy_estimator"] = "all actual finite mixture components; exact density, sampled entropy expectation"
                if cfg.alg.actor.get("soft_proximal_ess_fraction", 0.0) > 0:
                    environment_metadata.update(
                        teacher="actual finite policy conditional mixture; no teacher-only std floor",
                        extraction="KL-proximal soft target; ESS selects eta multiplying Q/T and -log pi together",
                        objective_temperature=float(cfg.alg.actor.temperature),
                    )
            if cfg.alg.actor.get("teacher_distribution") == "best_of_k_winners":
                winner_count = int(cfg.alg.actor.num_policy_samples * cfg.alg.actor.proposals_per_policy_sample)
                environment_metadata.update(
                    teacher="independent best-of-k winners from current full Gaussian actor; live twin-min",
                    teacher_objective="winner_distribution", teacher_best_of_k=int(cfg.alg.actor.teacher_best_of_k),
                    teacher_winner_count=winner_count,
                    teacher_candidate_count=winner_count * int(cfg.alg.actor.teacher_best_of_k),
                    teacher_weights="uniform", teacher_density_correction=False,
                    teacher_std_floor=0.0, teacher_bandwidth_space=None,
                    teacher_temperature=None, teacher_pretanh_targets="original sampled u; stop-gradient",
                )
            if cfg.alg.actor.get("proposal_best_of_k", 1) > 1:
                environment_metadata.update(
                    teacher="independent best-k pilot guided Gaussian mixture with original proposal coverage",
                    teacher_objective="Boltzmann exp(Q/T); self-normalized importance sampling",
                    teacher_q_aggregation=cfg.alg.actor.source_q_eval,
                    teacher_density_correction="exact conditional proposal mixture density, beta=1",
                    proposal_best_of_k=int(cfg.alg.actor.proposal_best_of_k),
                    proposal_guided_fraction=float(cfg.alg.actor.proposal_guided_fraction),
                    proposal_pilot_count=int(cfg.alg.actor.num_policy_samples * cfg.alg.actor.proposal_best_of_k),
                    teacher_pretanh_targets="fresh independent proposal draws; original u retained",
                )
            if model.behavior_best_of_k > 1:
                environment_metadata["collection"] = {
                    "best_of_k": model.behavior_best_of_k,
                    "candidates": "current actor; independent z and Gaussian epsilon",
                    "selection": "argmax of live min(Q1,Q2)",
                    "starts_after_uniform_warmup": int(cfg.alg.learning_starts),
                    "td_target_best_of_k": False,
                    "evaluation_best_of_k": False,
                }
            if cfg.get("dual_mu_eval", False):
                environment_metadata["evaluation"] = {
                    "zero_z": "a=tanh(mu(s,0)); epsilon=0",
                    "stochastic_z": "z~N(0,I) per action; a=tanh(mu(s,z)); epsilon=0",
                    "episodes_per_mode": int(cfg.num_eval_episodes),
                    "legacy_eval_alias": "zero_z",
                    "paired_episode_reset_seeds": True,
                    "rng_isolated_from_collection": True,
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
            if isinstance(evaluation, DualMuEvalCallback):
                for mode, history in evaluation.histories.items():
                    values = history["results"][-1]
                    run.summary[f"final_eval_return_{mode}"] = float(sum(values) / len(values))
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


@hydra.main(version_base=None, config_path="configs", config_name="mujoco_v5")
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
