# Spline Energy Circuit: one model for Q, V and sampling

User request, 2026-09-21: develop a distinctive unified policy/Q architecture and
actually test it on GMM40; prioritize an idea over a suite of ablations.

## Hypothesis and construction

Use a positive sum-product circuit on the box, rather than Gaussian conditionals
whose centers need a saturating action transform. Each leaf f_rj is a positive
piecewise-linear function on [-1,1], parameterized by log heights at 129 fixed
knots. The rank is 64. The only learned objects are these leaves and root heights.

    F(a) = sum_r exp(h_r) product_j f_rj(a_j)
    Q(a) = log F(a)
    I_rj = integral_-1^1 f_rj(x) dx
    Z = sum_r exp(h_r) product_j I_rj
    V = log Z
    pi(a) = F(a) / Z

Leaf integrals are exact trapezoid sums because the interpolated density, not its
logarithm, is piecewise linear. Component probabilities follow from integrated
energy mass, rather than a separately learned actor. One ancestral sample chooses
a root and, in parallel for all coordinates, an interval and the analytic inverse
of its linear density CDF. No diffusion/ODE/MCMC, learned Gaussian sigma, tanh, or
external Q selection is used. Categorical CDF search is arithmetic, not an
iterative neural sampler. Stochastic samples are the policy evaluated below.

This is a trainable sum-product network. In this state-free toy there is no
state-conditioning MLP, and we do not claim that MuJoCo generalization or
state-conditioned inference speed has been demonstrated. A future conditional
version would emit these parameters from a state encoder in one forward.

Related foundations: probabilistic circuits support tractable sum/product
inference (Gala et al., AISTATS 2024,
https://proceedings.mlr.press/v238/gala24a.html). IQ and MEow motivate obtaining a
policy from the same energy representation. This prototype's box spline leaves,
analytic ancestral sampling and energy-fitting protocol are a proposed
combination, not a claim that probabilistic circuits or exact normalization are
new discoveries.

## Learning the external Q

Set Q_target=log p_GMM40 and temperature=1. Each update makes 512 target-value
queries, with no target gradient and no target samples in training. The proposal
is b = 0.5*pi_old + 0.5*Uniform(box), ensuring support on the full box. Stop the
proposal, actions and importance weights during differentiation.

    L = Z_theta - mean[ exp(Q_target(a))/b(a) * Q_theta(a) ], a~b

This is the generalized-KL / Poisson energy-fitting objective, up to a
theta-independent target term. Its unconstrained population optimum is
F_theta=exp(Q_target), so both the energy level and its normalized policy are
learned. The integral of the model energy is exact; the target-weighted term is
Monte Carlo. Weights are not self-normalized. It is related to density fitting;
we do not claim this algebra eliminates approximation or importance variance.

A uniform energy shift changes Z and is learned by this objective. This differs
from claiming success because pi=exp(Q-V), which is an identity. Metrics always
use the independently specified external GMM40 target. The objective and
defensive proposal differ from Direct SNIS by design.

## Experiment and comparison

Main run: 20,000 updates, seeds 0..4, Adam 3e-4, 512 target calls/update,
10.24 million target value calls per seed. Same target parameters, action box,
coordinate scale 50, evaluation RNGs, sample count 16,384, forward/reverse KL,
strict 3-sigma mode coverage, radial 3-sigma precision and nearest-cell mass TV
as existing Direct/SMEM experiments.

Initialization uses a target-independent 8x8 covering grid of local leaves,
initial width .25, center jitter .01 and log-height jitter .02. Leaf integrals
are normalized to equal initial root mass and total Z=1. Width is initialization,
not a trainable sigma or a hard constraint. This differs from the legacy
Direct's near-origin initialization; report step-zero results and do not claim
a single-variable causal comparison. Circuit has 16,576 trainable parameters.

Baseline: preserve and reuse smem_compare_snis_seed{0..4}_a1, 20k updates,
source 1213f1496b8c45036657c0cad88da2b4f61af691. It has the same target-value
query budget and evaluation metric definitions, but a different architecture,
initialization, loss and proposal. This is a whole-method comparison, not an
architecture-only ablation. Historical training-time comparisons are descriptive.
Same-device batch-1 samplers are timed separately, including circuit CDF setup
and normalization, with no cached or constant-folded parameters.

Changing the model family removes Gaussian sigma parameters. Direct's default
log_std [-5,-1] is untouched, as are all existing snapshots and outputs.

## Informative validation and provenance

Before launch, validate exact integral against a 129x129 grid quadrature (exact
for this piecewise bilinear surface), sample bounds and joint 8x8 histogram
against integrated model cell probabilities using 65,536 samples, and three
finite nonzero-gradient updates. No extra testing framework or hyperparameter
sweep is introduced.

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      ../.venv/bin/python validate_spline_energy.py
    ../.venv/bin/python launch_spline_energy.py \
      --phase spline_energy_v1 --steps 20000 --seeds 5 --attempt a1

Validate source hashes; commit all source and protocol before launch. Launcher
creates an immutable Git snapshot, rejects duplicate output/receipt/job names,
records live cluster diagnostics, command, full SHA and job ID in a sidecar.
Jobs request one GPU and four CPUs, base_qos for these short toy runs, 15 minutes.
W&B uses OptiQ/OptiQ-GMM40-Sampling-Comparison, group spline_energy_v1, with
algorithm=Spline Energy Circuit for grouping.

## Limits and next inference

Success on a fixed state-free 2D target would validate the representation and
sampler, not TD stability, high-dimensional circuit rank, an arbitrary target Q,
or one-forward MuJoCo performance. Marginal splines can express multiple peaks;
the sum of products supplies cross-coordinate dependence, subject to finite rank.
General deterministic argmax is not automatically tractable in high dimensions.
The current test measures full stochastic sampling, without deleting noise or
selecting high-Q samples at evaluation.

## Post-training sampler implementation update

All five original training runs finished at commit
628d1244d0b7d1fc23c9a1783be7dde3938f8eae. The original frozen source remains
unchanged. The follow-up parallel_root sampler replaces binary CDF search by a
parallel count of cumulative probabilities below a uniform variate. It is the
same inverse categorical CDF, not a new policy. Exact sample equality is checked
on 8,192 actions for each of the five trained checkpoints. A separate one-GPU
benchmark compares both implementations and Direct on the same hardware; it
does not retrain any model. The benchmark has its own code commit and receipt.
The post-launch report script is separate from the training source provenance.
