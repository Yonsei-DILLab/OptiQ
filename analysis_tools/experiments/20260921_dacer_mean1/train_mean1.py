"""HalfCheetah/Ant DACER with Xavier mean-head initialization only."""
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "mean1_prior_dacer", HERE.parent / "20260921_dacer_priority/train_dacer.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
train = prior.train
from omegaconf import OmegaConf

CAMPAIGN = "trg-dacer-mean1-20260921"


def name(task, seed):
    return f"{task}-trg-dacer-mean1-T0.25-b1-s{seed}"


def compose_config(task, seed, output):
    assert task in ("halfcheetah", "ant")
    assert seed in (range(2) if task == "halfcheetah" else range(4))
    cfg = train.compose_config([
        f"benchmark={task}", f"seed={seed}",
        "alg.actor.mean_output_init_scale=1.0", "alg.actor.temperature=0.25",
        "alg.actor.density_beta=1", "alg.actor.density_correction_beta=1",
        "dacer.enabled=true", f"dacer.noise_scale={.15 if task == 'halfcheetah' else .1}",
        "wandb.project=gmm-trg", "wandb.job_type=dacer-mean-init",
        f"wandb.group={CAMPAIGN}-{task}-T0.25-b1", f"run_name={name(task, seed)}",
        f"output_root={output}", "require_gpu=true",
        f"+experiment.campaign={CAMPAIGN}", "+experiment.stage=dacer-mean-init",
    ])
    reference = train.base.compose_config([f"benchmark={task}"])
    reference.alg.actor.mean_output_init_scale = 1.0
    assert OmegaConf.to_container(cfg.alg, resolve=True) == OmegaConf.to_container(reference.alg, resolve=True)
    for key in ("total_steps", "eval_interval", "num_eval_episodes", "dual_mu_eval", "mu_only_eval"):
        assert cfg[key] == reference[key]
    assert cfg.total_steps == 1_000_000 and cfg.alg.batch_size == 256 and cfg.alg.utd == 1
    assert cfg.alg.optimizer.lr_actor == cfg.alg.optimizer.lr_critic == 3e-4
    assert cfg.alg.actor.log_std_min == -5 and cfg.alg.actor.log_std_max == -1
    assert cfg.alg.actor.initial_log_std == -1
    assert cfg.dacer.target_entropy_per_dim == -.9
    assert cfg.dual_mu_eval and cfg.mu_only_eval
    return cfg


if __name__ == "__main__":
    task, seed, output = sys.argv[1:]
    cfg = compose_config(task, int(seed), output)
    gpu = int(os.environ["CAMPAIGN_GPU"])
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[gpu::4] or cpus)
    original = train.runner.create_algorithm
    def create(config):
        model, callbacks = original(config)
        callbacks.callbacks.append(prior.Metadata())
        return model, callbacks
    train.runner.create_algorithm = create
    train.compose_config = lambda unused: cfg
    train.main()
