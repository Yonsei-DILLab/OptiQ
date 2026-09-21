"""Targeted CPU validation for the state-conditioned SMEM+TR transfer."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm

import train
from optiq_dime.smem_tr import box_kl, project


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def quadrature_kl(mu, log_std, ref_mu, ref_log_std):
    std, ref_std = np.exp(log_std), np.exp(ref_log_std)
    a, b = (-1.-mu)/std, (1.-mu)/std
    ra, rb = (-1.-ref_mu)/ref_std, (1.-ref_mu)/ref_std
    def integrand(x):
        log_p = truncnorm.logpdf(x, a, b, loc=mu, scale=std)
        log_q = truncnorm.logpdf(x, ra, rb, loc=ref_mu, scale=ref_std)
        return np.exp(log_p)*(log_p-log_q)
    return quad(integrand, -1., 1., epsabs=1e-9)[0]


def checks():
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Validation must run with JAX_PLATFORMS=cpu")
    result = {}
    errors = []
    for mu, log_std, ref_mu, ref_log_std in (
        (.98, -1.05, -.2, -1.3),
        (-.999, -5., -.98, -4.),
        (.4, -4.9, -.5, -4.8),
    ):
        actual = float(box_kl(
            jnp.array([mu]), jnp.array([log_std]),
            jnp.array([ref_mu]), jnp.array([ref_log_std]),
        ))
        expected = quadrature_kl(mu, log_std, ref_mu, ref_log_std)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)
        errors.append(abs(actual-expected))
    result["kl_quadrature_absolute_errors"] = errors

    old = (jnp.array([[-.6], [.92]]), jnp.array([[-2.], [-1.1]]))
    candidate = (jnp.array([[-.25], [.55]]), jnp.array([[-2.5], [-2.]]))
    projected, fraction = jax.jit(project)(old, candidate, .03)
    projected_kl = float(box_kl(*projected, *old).mean())
    assert .02999 < projected_kl < .030002
    assert 0. < float(fraction) < 1.
    result["projection"] = {
        "joint_kl": projected_kl, "fraction": float(fraction)
    }

    configs = {}
    for method in ("direct", "smem_tr"):
        cfg = train.compose_config(["benchmark=halfcheetah"], method)
        configs[method] = {
            "env": cfg.env_name,
            "steps": int(cfg.total_steps),
            "seed": int(cfg.seed),
            "loss": cfg.alg.actor.distillation_loss,
            "log_std": [float(cfg.alg.actor.log_std_min), float(cfg.alg.actor.log_std_max)],
        }
    assert configs["direct"]["loss"] == "direct_gmm_nll"
    assert configs["smem_tr"]["loss"] == "smem_tr"
    assert configs["direct"]["env"] == configs["smem_tr"]["env"] == "HalfCheetah-v4"
    assert configs["direct"]["steps"] == configs["smem_tr"]["steps"] == 1_000_000
    result["configs"] = configs

    cfg = train.compose_config([
        "benchmark=halfcheetah", "seed=0", "alg.batch_size=2",
        "alg.buffer_size=32", "alg.learning_starts=2",
        "alg.actor.learning_starts=2", "num_eval_episodes=1",
        "checkpoint_interval=0", "diagnostic_interval=0",
        f"output_root={tempfile.mkdtemp(prefix='hc-smem-validation-')}",
    ], "smem_tr")
    model, callbacks = train.runner.create_algorithm(cfg)
    actor = cfg.alg.actor
    observations = jnp.zeros(
        (2, int(np.prod(model.observation_space.shape))), dtype=jnp.float32
    )
    try:
        state, _, _, metrics = model.update_actor(
            model.policy.actor_state, model.policy.qf_state, observations,
            model.key,
            jnp.linspace(cfg.alg.critic.v_min, cfg.alg.critic.v_max, cfg.alg.critic.n_atoms),
            actor.num_policy_samples, actor.proposals_per_policy_sample,
            actor.proposal_sampling_mode, actor.proposal_std, actor.proposal_clip,
            actor.include_anchor, actor.density_correction, actor.density_beta,
            actor.adaptive_density_beta, actor.minimum_source_ess,
            actor.density_beta_grid_size, actor.temperature,
            actor.sinkhorn_epsilon, actor.sinkhorn_iterations,
            actor.source_q_eval, actor.transport_target_mode, True, False,
            actor.distillation_loss, actor.teacher_distribution,
            actor.get("soft_proximal_ess_fraction", 0.), False,
            actor.ot_student_action,
        )
        jax.block_until_ready(state)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves((state, metrics)))
        assert float(metrics["teacher_nll_gain"]) >= -1e-6
        assert float(metrics["teacher_kl_bound"]) <= .050002
        assert float(metrics["actor_nll_gain"]) >= -1e-6
        assert float(metrics["actor_kl_bound"]) <= .050002
        assert int(state.step) == int(model.policy.actor_state.step)+1
        names = (
            "old_nll", "teacher_nll_gain", "teacher_kl_bound",
            "smem_attempted", "smem_selected_state_fraction",
            "actor_nll_gain", "actor_kl_bound", "actor_kl_state_max",
            "actor_accepted", "final_actor_rejected", "projection_loss",
            "actor_std_mean",
        )
        result["halfcheetah_actor_update"] = {
            name: float(metrics[name]) for name in names
        }
    finally:
        callbacks.callbacks[0].eval_env.close()
        model.get_env().close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = [HERE/"optiq_dime"/name for name in (
        "smem_tr.py", "algorithm.py", "policy.py", "box_gaussian.py"
    )] + [HERE/"train.py", Path(__file__)]
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "output": str(args.output.resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).splitlines(),
        "source_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "optax", "numpy", "scipy", "gymnasium", "mujoco")
        },
    }
    try:
        record.update(checks(), passed=True)
    except BaseException as exc:
        record.update(passed=False, failure=repr(exc))
        args.output.write_text(json.dumps(record, indent=2)+"\n")
        raise
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
