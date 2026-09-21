"""Bounded exact-target probe of the common v2 actor projection (no RL).

Q = T log p for an equally weighted two-component tanh Gaussian. This
isolates projection capacity: no critic learning, replay, or soft acceptance
guard is used. Good toy results do not establish Humanoid performance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gymnasium import spaces
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from optiq_dime import OptiQDIME
from optiq_dime.policy import OptiQPolicy
from optiq_dime.latent import FiniteMixtureTrainState, finite_latent_codes

TEMPERATURE = .1
TARGET_STD = .15
VARIANTS = {
    "current": {"initial_log_std": float(np.log(.5)), "log_std_max": 1.,
                "mean_output_init_scale": 1.e-4},
    "cap02": {"initial_log_std": float(np.log(.2)), "log_std_max": float(np.log(.2)),
              "mean_output_init_scale": 1.e-4},
    "cap02_init01": {"initial_log_std": float(np.log(.2)), "log_std_max": float(np.log(.2)),
                     "mean_output_init_scale": .1},
    "latent_skip": {"initial_log_std": float(np.log(.5)), "log_std_max": 1.,
                    "mean_output_init_scale": 1.e-4, "diagnostic_latent_skip_scale": 1.},
    "latent_skip_cap02": {"initial_log_std": float(np.log(.2)), "log_std_max": float(np.log(.2)),
                          "mean_output_init_scale": 1.e-4, "diagnostic_latent_skip_scale": 1.},
}


def exact_q(variables, obs, actions, **kwargs):
    # Clipping only protects atanh against float32 tanh rounding at the boundary.
    a = jnp.clip(actions[:, 0], -1. + 1.e-6, 1. - 1.e-6)
    u = jnp.arctanh(a)
    component_logp = -.5 * ((u[:, None] - jnp.array([-1., 1.])) / TARGET_STD)**2
    component_logp -= jnp.log(TARGET_STD * jnp.sqrt(2. * jnp.pi))
    logp = jax.scipy.special.logsumexp(component_logp, axis=-1) - jnp.log(2.)
    logp -= jnp.log1p(-a*a)
    return jnp.broadcast_to((TEMPERATURE * logp)[None, :, None], (2, len(obs), 1))


def target_density(a):
    if abs(a) >= 1.:
        return 0.
    u = np.arctanh(a)
    return .5 * (norm.pdf(u, -1., TARGET_STD) + norm.pdf(u, 1., TARGET_STD)) / (1.-a*a)


def evaluate(actor, seed, samples):
    # Fixed independent evaluation randomness across snapshots/variants.
    rng = np.random.default_rng(1600000 + seed)
    if isinstance(actor, FiniteMixtureTrainState):
        codes = np.asarray(finite_latent_codes(actor, 1))
        z = codes[rng.integers(actor.latent_components, size=samples)]
    else:
        z = rng.normal(size=(samples, 1)).astype(np.float32)
    eps = rng.normal(size=(samples, 1)).astype(np.float32)
    mu, log_std = actor.apply_fn({"params": actor.params}, jnp.zeros_like(z), jnp.asarray(z))
    mu, std = np.asarray(mu, dtype=np.float64), np.exp(np.asarray(log_std, dtype=np.float64))
    actions = np.tanh(mu + std*eps).reshape(-1)
    # Deterministic component quantiles: true component overlap is < 3e-11.
    half = samples//2
    quantiles = norm.ppf((np.arange(half)+.5)/half)
    target = np.sort(np.tanh(np.concatenate((-1.+TARGET_STD*quantiles, 1.+TARGET_STD*quantiles))))
    between, within = float(mu.var()), float(np.mean(std**2))
    return {
        "w2_squared": float(np.mean((np.sort(actions)-target)**2)),
        "negative_mode_fraction": float(np.mean(actions < -.5)),
        "positive_mode_fraction": float(np.mean(actions > .5)),
        "central_fraction": float(np.mean(np.abs(actions) <= .5)),
        "latent_mean_variance_fraction": between/(between+within),
        "mu_std_over_z": float(mu.std()), "conditional_std_mean": float(std.mean()),
        "log_std_std_over_z": float(np.log(std).std()),
        "boundary_fraction": float(np.mean(np.abs(actions) >= 1.-1.e-6)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--config", default="mujoco_v2_checked")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=["current", "cap02", "cap02_init01"])
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/v2_improvement/bimodal_projection.json")
    args = parser.parse_args()
    assert args.updates > 0 and args.samples >= 1024 and args.samples % 2 == 0
    assert len(set(args.seeds)) == len(args.seeds)
    normalization, error = quad(target_density, -1., 1., epsabs=1.e-10)
    assert abs(normalization-1.) < 1.e-8 and error < 1.e-7
    # Cross-check JAX critic units/Jacobian against a separate scalar density.
    probe = np.array([-.9, -.5, 0., .5, .9], dtype=np.float32)
    q_probe = np.asarray(exact_q({}, np.zeros((5, 1)), jnp.asarray(probe[:, None])))[0, :, 0]
    np.testing.assert_allclose(q_probe, TEMPERATURE*np.log([target_density(a) for a in probe]), atol=1.e-5)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "devices": [str(d) for d in jax.devices()], "updates": args.updates,
        "seeds": args.seeds, "evaluation_samples": args.samples, "batch_size": 64, "config": args.config,
        "target": {"pretanh_means": [-1., 1.], "pretanh_std": TARGET_STD,
                   "mixture_weights": [.5, .5], "density_integral": normalization,
                   "central_fraction": quad(target_density, -.5, .5, epsabs=1.e-12)[0]},
        "variants": {k: VARIANTS[k] for k in args.variants}, "rows": [], "complete": False,
        "limitations": ["Projection-only; no critic updates or soft guard.",
            "1D synthetic target is not a Humanoid result or an improvement guarantee.",
            "Q uses atanh boundary clipping at 1-1e-6.",
            "W2 uses finite fresh actor samples and deterministic target component quantiles."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)

    save()  # Freeze the bounded protocol before collecting results.
    started = time.monotonic()
    for name in args.variants:
        overrides = VARIANTS[name]
        for seed in args.seeds:
            with initialize_config_dir(version_base=None, config_dir=str(ROOT/"configs")):
                cfg = compose(config_name=args.config)
            for key, value in overrides.items():
                if key != "diagnostic_latent_skip_scale":
                    setattr(cfg.alg.actor, key, value)
            policy = OptiQPolicy(spaces.Box(-1., 1., (1,), dtype=np.float32),
                                spaces.Box(-1., 1., (1,), dtype=np.float32), cfg)
            policy.build(jax.random.PRNGKey(seed), lambda _: .0003, .0003)
            actor = policy.actor_state
            if "diagnostic_latent_skip_scale" in overrides:
                # Probe an explicit residual mean initialization without changing
                # repository policy/training code. Still one Gaussian G(s,z).
                base_apply = actor.apply_fn
                scale = overrides["diagnostic_latent_skip_scale"]

                def residual_apply(variables, obs, z, base=base_apply, coefficient=scale):
                    mu, log_std = base(variables, obs, z)
                    return mu + coefficient*z, log_std

                actor = actor.replace(apply_fn=residual_apply)
            critic = policy.qf_state.replace(apply_fn=exact_q)
            key = jax.random.PRNGKey(1500000+seed)
            observations = jnp.zeros((64, 1), dtype=jnp.float32)
            metrics = {}
            for step in range(args.updates+1):
                if step == 0 or step % 500 == 0 or step == args.updates:
                    row = {"variant": name, "seed": seed, "update": step,
                           **evaluate(actor, seed, args.samples), **metrics}
                    assert all(np.isfinite(v) for k, v in row.items() if k != "variant")
                    report["rows"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
                if step == args.updates:
                    break
                actor, _, key, update_metrics = OptiQDIME.update_actor(
                    actor, critic, observations, key, jnp.array([-3600.]),
                    num_policy_samples=16, proposals_per_policy_sample=4,
                    proposal_sampling_mode="exact", proposal_std=.05, proposal_clip=.5,
                    include_anchor=False, density_correction=True, density_beta=1.,
                    adaptive_density_beta=False, minimum_source_ess=16., density_beta_grid_size=257,
                    temperature=TEMPERATURE, sinkhorn_epsilon=.25, sinkhorn_iterations=100,
                    source_q_eval="mean", transport_target_mode="argmax", semi_implicit=True,
                    normalize_ot_cost=False, distillation_loss="conditional_ot_nll",
                    teacher_distribution="conditional_mixture")
                if (step+1) % 500 == 0 or step+1 == args.updates:
                    metrics = {k: float(update_metrics[k]) for k in
                               ("actor_loss", "source_ess_absolute", "policy_entropy_lower")}
    report.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic()-started)
    save()


if __name__ == "__main__":
    main()
