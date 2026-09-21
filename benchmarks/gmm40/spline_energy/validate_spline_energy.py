"""Only checks new analytic integration, inverse sampler and loss gradients."""
import hashlib, json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from core import target_parameters
from spline_energy import (initialize_circuit, circuit_q, circuit_partition,
                           circuit_sample, coarse_cell_mass, make_circuit_update)
ROOT = Path(__file__).resolve().parent

def main():
    state, key = initialize_circuit(991)
    # Irregular leaves catch errors masked by Gaussian/uniform special cases.
    params = dict(state.params)
    params["leaves"] = params["leaves"] + .5*jax.random.normal(key, params["leaves"].shape)
    params["height"] = params["height"] + jnp.linspace(-2., 2., 64)
    grid = jnp.linspace(-1., 1., 129)
    xy = jnp.stack(jnp.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
    q = np.asarray(jax.jit(circuit_q)(params, xy)).reshape(129,129)
    numerical_z = np.trapz(np.trapz(np.exp(q.astype(np.float64)), np.asarray(grid), axis=1),
                          np.asarray(grid), axis=0)
    analytic_z = float(jnp.exp(circuit_partition(params)[0]))
    relative_error = abs(numerical_z/analytic_z-1.)
    assert relative_error < 2e-5, relative_error
    sample = np.asarray(jax.jit(circuit_sample, static_argnums=2)(params, jax.random.PRNGKey(543), 65536))
    assert np.isfinite(sample).all() and (abs(sample)<=1.).all()
    hist = np.histogram2d(*sample.T, bins=8, range=[[-1,1],[-1,1]])[0]/len(sample)
    expected = np.asarray(coarse_cell_mass(params, 8))
    sampler_tv = .5*abs(hist-expected).sum()
    assert sampler_tv < .025, sampler_tv
    target = target_parameters()
    updater = make_circuit_update(target)
    initial = state
    infos = []
    for _ in range(3):
        state, key, info = updater(state, key)
        jax.block_until_ready(state)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
        infos.append({k:float(v) for k,v in info.items()})
    assert float(jnp.linalg.norm(state.params["leaves"]-initial.params["leaves"])) > 0
    receipt = dict(passed=True, analytic_integral_relative_error=relative_error,
        sampler_coarse_tv=sampler_tv, sample_count=65536, updates=infos,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                       for f in ("spline_energy.py","validate_spline_energy.py")})
    out = ROOT.parent/"manifest"/"spline_energy_validation.json"
    out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))

if __name__ == "__main__":
    main()

