# K=3 mean-only: a deliberately initialized bad local minimum

## Question and relation to the earlier experiments

The previous `finite_gmm_six_1d` paper control used target means
`[-4.25, 0, 4.25]`, sigma 0.5, equal weights, and plain GD for 100,000
updates. All 12 runs, including the four bad-structure initializations,
converged to the true means. That is an unsuccessful reproduction of a
bad local minimum, not evidence for one.

Here we follow the asymmetric construction in Jin et al. (2016),
*Local Maxima in the Likelihood of Gaussian Mixture Models*, section 4.1.1:
<https://arxiv.org/pdf/1609.00978>. Two target components are relatively
near each other, while a third is far away. One fitted component covers
the near pair, and two fitted components explain the far component.
This is an illustrative deliberately initialized reproduction. It does
not estimate the probability of failure from random initialization.

## Exact statistical model

Both target and student are ordinary, unbounded, one-dimensional GMMs:

\[
p^*(x)=\tfrac13\sum_{k=1}^3\mathcal N(x;c_k,0.5^2),\qquad
q_\mu(x)=\tfrac13\sum_{i=1}^3\mathcal N(x;\mu_i,0.5^2).
\]

Only the three means are trainable. Weights and sigma stay fixed.
The objective is population NLL, or equivalently forward KL:

\[
\mathcal L(\mu)=-\int p^*(x)\log q_\mu(x)\,dx,\qquad
\mu\leftarrow\mu-0.01\nabla_\mu\mathcal L.
\]

There is no implicit actor, latent resampling, SNIS, stochastic target
batch, action bound, truncation, Adam, learned weight or variance. This
isolates finite-GMM optimization geometry and is intentionally distinct
from the bounded-action semi-implicit actor study.

## Predeclared cases and initialization

Targets are `[-R, R, D]` for R in {1, 1.5} and D in {4, 5, 6, 8}.
The old symmetric target is a ninth control. Each case contains four
paired seeds for each initialization:

- Bad structure: `[(c0+c1)/2, c2-0.2, c2+0.2]` plus iid Gaussian jitter
  with standard deviation 0.05.
- Good structure: true means plus the same seed's jitter.

Thus 9 targets x 2 initializations x 4 seeds = 72 trajectories. All run
100K updates; every 1K we save parameters, population NLL and gradient
norm. The eight trajectories within a job are independently vectorized,
with no parameter sharing.

## Numerical checks and evidence standard

Training uses float64 and a 4,097-point Simpson rule, from the leftmost
target mean minus 10 sigma to the rightmost mean plus 10 sigma. This is
integration of an unbounded model, not a truncated student distribution.
Validation compares autodiff gradients and Hessians to independent
adaptive integration of their analytic expressions and finite differences.

For every endpoint, record the loss gap to the representable target,
gradient norm, and Hessian eigenvalues. Refine integration to 16,385
points and independently verify candidate bad minima with adaptive
integration. A small gradient alone, or persistence for 100K steps, is
not sufficient: a saddle can escape very slowly.

Newton/root polishing, if used, is strictly a stationary-point diagnostic;
record both raw GD parameters and the polishing displacement. It does
not replace the training endpoint. A strict bad minimum requires positive
loss gap and positive curvature clearly above quadrature disagreement.
Report flat/uncertain cases separately. Check objective changes under
small perturbations along Hessian eigenvectors and random directions.

Select an illustrative figure only among numerically verified strict bad
minima, preferring a case where all four bad-start seeds reproduce it,
then the largest minimum Hessian eigenvalue. Preserve a table of all nine
cases, including successful escapes. Sampling evaluation uses 2^20 actions
and 512 histogram bins; the histogram range is the integration interval.

## Execution and reproducibility

Commit this package to `heejoon` before launch. Store full SHA and source
SHA256 hashes in SOURCE_MANIFEST.json and record Slurm job IDs. CPU-only
Slurm jobs use two cores each; this three-parameter calculation does not
need a GPU. Results and checkpoints belong on login4 and dildata, not Git.
