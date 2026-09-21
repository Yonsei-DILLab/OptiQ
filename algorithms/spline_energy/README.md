# Spline Energy

The restored `--profile gmm_reference` uses [`raw_energy.py`](raw_energy.py):
the GMM40 unnormalized circuit, rank 64 / 129 knots, derived log partition,
and ordinary TD + MSE checked against Direct GMM/TRG commit `30f4db1`.
See [`DIRECT_ALIGNMENT.md`](DIRECT_ALIGNMENT.md) for equations, reference-code
mapping, the self-imitation issue, and differences from the separate-critic
Direct implementation. The native one-forward sampler is retained.

The description below documents the earlier normalized-value parameterization.

`model.py` contains the algorithm's shared state encoder, exact normalized
positive-spline policy, tied soft Q function, and one-forward action sampler.

For a state `s`, the circuit outputs a scalar `V(s)`, mixture weights `w_r(s)`,
and normalized one-dimensional positive linear splines `f_rd(a_d | s)`:

```text
pi(a | s) = sum_r w_r(s) prod_d f_rd(a_d | s)
Q(s, a)   = V(s) + alpha log pi(a | s)
```

Consequently, `integral exp(Q(s,a) / alpha) da = exp(V(s) / alpha)` exactly on
the normalized action box. `sample_action` uses a categorical root CDF and
parallel analytic inverse CDFs for the selected spline leaves. It performs one
network forward pass and has no diffusion loop, MCMC, optimization, or action
candidate search at inference time.

The committed MuJoCo training and validation harness is under
`analysis_tools/experiments/20260921_mujoco_spline_energy/`.

The original state-free GMM40 implementation, exact 100k comparison runner,
target definition, and five-seed measurements are under
`benchmarks/gmm40/spline_energy/`.

The opt-in `--loss-kind relative_energy` trainer keeps this model unchanged and
replaces Huber regression with a target-relative energy divergence. See
[`THEORY.md`](THEORY.md) for its GMM40 derivation, Bellman residual/performance
bounds, stochastic-target bias, numerical continuation, and explicit limits.
[`energy_bellman.py`](energy_bellman.py) implements the scalar loss and conditional
bound calculators. The existing 15-run Huber campaign uses its original snapshot.

The [Ant 100k pilot](PILOT_20260922.md) did not establish an improvement:
returns remained negative and sampled TD errors increased. This remains an
opt-in research variant, not a replacement for the validated GMM40 reference.
