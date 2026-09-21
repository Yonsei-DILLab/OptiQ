# HalfCheetah Direct GMM/TRG versus SMEM+TR

## Question and comparison

Hypothesis: the narrow, mode-seeking components produced by the GMM40
SMEM+TR update may improve control return when transferred to the bounded
64-component HalfCheetah policy.  The baseline is the committed bounded
Direct GMM/TRG implementation.  The only experimental variable is the actor
fitting update.  Both methods use `HalfCheetah-v4`, seeds 0/1/2, one million
environment steps, identical initialization, 5K warmup, replay, batches,
critic, optimizer, temperature 0.25, 64 components/candidates, and log-standard
deviation bounds `[-5,-1]`.

Primary metrics are stochastic-z and zero-z evaluation return at one million
steps (10 paired-reset episodes per mode) and evaluation-return AUC.  Report
actor scale, acceptance, joint-KL, runtime, and the full learning curves as
diagnostics.  A three-seed result is measured evidence for this implementation,
not a claim of broad MuJoCo superiority.

## Direct GMM/TRG baseline

For every replay state, draw 64 fresh normal latents and obtain 64 normalized
box-truncated diagonal Gaussians.  Draw 64 IID candidates from their uniform
mixture.  With the exact normalized proposal density, form stopped weights

```
w_j = softmax(Q_mean(s,a_j)/0.25 - log q_old(a_j|s)).
```

The baseline takes one Adam step on the weighted marginal mixture NLL.  It
uses no OT, resampling, entropy backup, gradient clipping, or added uniform
behavior.

## State-conditioned SMEM+TR transfer

For each replay state independently, fit the stopped candidates and weights
with the same fixed-weight generalized EM used in the GMM40 experiment: two EM
iterations, four backtracked generalized M-steps, component ESS threshold 2,
and exact truncated-Gaussian normalization.  Every 20 outer actor updates,
search two overlap-ranked merge candidates and split a distinct component by
the local finite-sample mismatch score.  Run two partial-EM iterations on the
three affected slots, choose the least-cost of their six permutations, and
finish with full EM.  Component weights remain `1/64`.

Project each state's fitted teacher along the common natural-parameter path
until

```
(1/64) sum_k KL(T_new,k || T_old,k) <= 0.05,
```

where every KL is analytic for the normalized Gaussian conditioned on the
action box.  The joint component/action KL upper-bounds marginal-mixture KL.

The actor has shared parameters and fresh state-dependent latent components,
so fitted component slots cannot persist as free parameters.  Regress the
shared actor at the same state/latent pairs for up to 10 Adam steps.  Retain
the best iterate only when its weighted batch NLL does not increase and its
joint KL, averaged over replay states and components, is at most 0.05; otherwise
atomically restore parameters and Adam state.  Log the maximum per-state KL,
but do not constrain it.  This batch-average trust region is the deliberate RL
extension of the single-state GMM40 constraint.

SMEM+TR is an adaptation rather than the original variable-weight SMEM method:
mixture weights stay uniform, the bounded M-step is generalized rather than
closed form, search is periodic and limited, and the trust-region teacher is
stopped.  The actor-regression objective is auxiliary; no return guarantee
follows from improving its fixed weighted empirical NLL.

## Execution

Run the committed CPU validation, then a real-GPU 5K-warmup/200-update smoke
for each method.  If both are finite, launch the six independent one-GPU runs
under `big_qos`; the array has no cross-method dependency.  Preserve configs,
source commit, commands, SLURM IDs, outputs, evaluation archives, W&B URLs,
timing, and failure state under the campaign directory.
