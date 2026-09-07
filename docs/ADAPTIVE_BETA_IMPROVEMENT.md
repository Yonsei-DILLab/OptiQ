# Adaptive beta with a candidate expected-Q improvement constraint

This branch changes only the KL-mode budget from `KL(r || u)` to `KL(u || r)`.
Here `u` is uniform over the fixed candidate set and `r = softmax(Q / tau)`.
Existing fixed-beta and minimum-ESS modes, sampling, critic learning, and sweep
defaults are unchanged. Existing server runs are not migrated by this change.

## Rule and computation

For a state and M fixed candidates, let

\[
x_i=Q_i/\tau,\quad d_i=-\log q(a_i),\quad
u_i=1/M,\quad r=\operatorname{softmax}(x),\quad
w_\beta=\operatorname{softmax}(x+\beta d).
\]

Choose

\[
\beta^*=\max\{\beta\in[0,1]:
 D_{\mathrm{KL}}(w_\beta\Vert r)\le D_{\mathrm{KL}}(u\Vert r)\}.
\]

The right-hand side is `logsumexp(x) - mean(x) - log(M)`. Production code
evaluates it using the cancellation-resistant exponential-tilt KL helper,
starting at `r` and applying `-x`, to retain accuracy when Q is almost flat
in float32. It must NOT start at uniform and apply `x`: that computes the old
budget `KL(r || u)`.

The left-hand side has derivative

\[
\frac{d}{d\beta}D_{\mathrm{KL}}(w_\beta\Vert r)
=\beta\operatorname{Var}_{w_\beta}(d)\ge0.
\]

The existing 32-iteration bisection selects the feasible lower endpoint, or
exactly beta=1 when full correction is feasible. Flat Q selects beta=0 unless
density is also constant, in which case all beta values give the same weights
and the implementation selects 1. No new critic evaluations or Q gradients
are required inside the bisection. Inputs must be finite and tau positive.

## Exact finite-candidate guarantee

For any weights w, the identity

\[
\mathbb E_w Q-\mathbb E_u Q
=\tau\left[D_{\mathrm{KL}}(w\Vert u)
 +D_{\mathrm{KL}}(u\Vert r)-D_{\mathrm{KL}}(w\Vert r)\right]
\]

shows that every feasible solution satisfies

\[
\mathbb E_w Q-\mathbb E_u Q
\ge\tau D_{\mathrm{KL}}(w\Vert u)\ge0.
\]

This guarantee is in exact arithmetic; regression tests check floating-point
agreement with explicit tolerances. The old direction defines a valid but
different KL budget and does not imply this bound.

The rule also solves

\[
\min_{w\in\Delta_M}D_{\mathrm{KL}}(w\Vert w_1)
\quad\text{subject to}\quad
\mathbb E_w Q-\tau D_{\mathrm{KL}}(w\Vert u)\ge\mathbb E_u Q.
\]

Its Lagrange multiplier eta yields beta=1/(1+eta); the limiting case beta=0
corresponds to a zero budget with nonconstant densities.

## Scope and limitations

- The baseline is the uniform distribution over these same candidates. With
  anchors, it includes the deterministic anchors as well as random proposals.
- This is not an exact statement about the population KDE or current actor.
  Finite sampling, OT hard assignment, actor fitting, and subsequent KDE
  smoothing can change the target distribution and its expected Q.
- Learned-critic score improvement does not imply true RL-return improvement,
  an ESS lower bound, multimodal coverage, or superiority to full correction.
- Candidate densities must be finite/positive on evaluated candidates. A local
  truncated proposal does not establish coverage of the full action box.
- Logs from the old `parameter-sweep` commit must retain their old-budget
  interpretation. Do not silently relabel or resume them as corrected runs.

## Enable and test

Enable the existing mode explicitly; it remains opt-in:

```bash
python run_optiq_dime.py --config-name parameter_sweep \
  alg.actor.adaptive_density_beta=true \
  alg.actor.adaptive_density_beta_mode=kl
```

The selector and the RL integration have separate regression suites:

```bash
python -m pytest -q tests/test_density_beta_kl.py
python -m pytest -q tests/test_dime_diagnostics.py
```

The integration suite requires the repository's full RL dependencies. Neither
command starts the existing remote training jobs or uploads to W&B.

Validation of this change (2026-09-07): 34 local selector/sampling/sweep tests
passed; the full 49-test suite passed on CPU in an isolated temporary checkout
using the existing DIME dependency environment. All 256 stored Dog 50K candidate
pools also passed the score-improvement check, with mean beta within 3.1e-9 of
the independent float64 reference. These are software checks, not new RL runs.
