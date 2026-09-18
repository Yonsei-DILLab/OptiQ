"""Validate finite-M entropy brackets where the latent marginal is analytic.

For mu(z)=c*z and constant conditional sigma, u is exactly N(0,c²+sigma²).
This isolates estimator overlap/dimension effects; it is not a learned-policy
performance test. Calls the same entropy estimator as the running algorithm.
"""
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.integrate import quad
from scipy.stats import norm
from optiq_dime.soft_improvement import entropy_bracket_sample


def analytic_actor(variables, observations, z):
    params = variables["params"]
    return params["latent_scale"]*z, jnp.full_like(z, params["log_std"])


@partial(jax.jit, static_argnames=("components",))
def draw(actor, key, components):
    return entropy_bracket_sample(actor, jnp.zeros((256, 1)), key, components)


def main():
    out = ROOT/"outputs/v2_improvement/entropy_overlap_diagnostic.json"
    results = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "entropy_source_sha256": hashlib.sha256((ROOT/"optiq_dime/soft_improvement.py").read_bytes()).hexdigest(),
               "samples": 4096, "seed": 1700000, "rows": [], "complete": False}
    for dimension in (1, 17):
        for latent_scale, sigma in ((0., .5), (1., .5), (1., .2)):
            variance = latent_scale**2+sigma**2
            std = np.sqrt(variance)
            jacobian_mean = quad(lambda u: norm.pdf(u, scale=std)*2.*(
                np.log(2.)-u-np.logaddexp(0., -2.*u)), -12.*std, 12.*std, epsabs=1.e-10)[0]
            exact_entropy = dimension*(.5*np.log(2.*np.pi*np.e*variance)+jacobian_mean)
            actor = TrainState.create(apply_fn=analytic_actor, tx=optax.sgd(0.), params={
                "mu": {"bias": jnp.zeros(dimension)}, "latent_scale": jnp.array(latent_scale),
                "log_std": jnp.array(np.log(sigma))})
            for components in (16, 64, 256):
                records = []
                for key in jax.random.split(jax.random.PRNGKey(1700000), 16):
                    _, u, lo, hi = draw(actor, key, components)
                    u = np.asarray(u, dtype=np.float64)
                    exact_samples = .5*np.sum(u*u/variance+np.log(2.*np.pi*variance), axis=-1)
                    exact_samples += np.sum(2.*(np.log(2.)-u-np.logaddexp(0., -2.*u)), axis=-1)
                    records.append(np.stack((np.asarray(lo), np.asarray(hi), exact_samples), axis=-1))
                values = np.concatenate(records)
                lower_error, upper_error = values[:, 0]-values[:, 2], values[:, 1]-values[:, 2]
                row = {"dimension": dimension, "latent_scale": latent_scale, "conditional_sigma": sigma,
                       "components": components, "exact_entropy": float(exact_entropy),
                       "lower_mean": float(values[:, 0].mean()), "upper_mean": float(values[:, 1].mean()),
                       "lower_error_mean": float(lower_error.mean()),
                       "lower_error_mc_se": float(lower_error.std(ddof=1)/np.sqrt(len(values))),
                       "upper_error_mean": float(upper_error.mean()),
                       "upper_error_mc_se": float(upper_error.std(ddof=1)/np.sqrt(len(values))),
                       "bracket_gap_mean": float((values[:, 1]-values[:, 0]).mean()),
                       "unchanged_policy_guard_mean_T01": float(.1*(values[:, 0]-values[:, 1]).mean())}
                assert all(np.isfinite(v) for v in row.values())
                if latent_scale == 0.:
                    assert np.max(np.abs(values[:, :2]-values[:, 2, None])) < 2.e-5
                results["rows"].append(row)
                out.write_text(json.dumps(results, indent=2))
                print(json.dumps(row), flush=True)
    results.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
