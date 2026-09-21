# GMM40 Spline Energy reference

This directory contains the state-free two-dimensional Spline Energy circuit
that motivated the conditional MuJoCo version.

- `spline_energy.py`: rank-64, 129-knot sum-product spline circuit, analytic
  normalization, density evaluation, Poisson energy fitting, and exact
  one-shot sampling.
- `train_spline_energy.py`: original self-contained 20k training program.
- `validate_spline_energy.py`: quadrature, sampler, and finite-gradient checks.
- `reference_100k_compare_latest.py`: exact frozen paired runner used for the
  reported 100k Spline Energy versus Direct GMM/TRG comparison. It retains the
  original campaign paths and upstream commit checks as provenance.
- `target_definition.json`: exact bounded DiKL GMM40 seed-0 target used by the
  100k comparison.
- `SPLINE_ENERGY.md`: registered equations, assumptions, and limits.
- `RESULTS.md`: measured five-seed results and source provenance.

The algorithm code is state-free here. The conditional version is in
`algorithms/spline_energy/model.py`; it emits the same normalized spline
parameters from a state encoder.

Run the cheap reference validation from this directory after creating
`../manifest`:

```bash
JAX_PLATFORMS=cpu python validate_spline_energy.py
```

The archived 100k runner is retained to identify exactly what produced the
reported results. It expects the original private upstream checkout at commit
`811e0e2e59a0c9137f38433e4fc18adaf5e6a8fa` and should not be launched as a new
experiment without changing its campaign paths.
