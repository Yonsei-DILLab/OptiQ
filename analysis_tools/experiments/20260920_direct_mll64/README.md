# Direct marginal likelihood: online 64 -> 64

User request, 2026-09-20: use heejoon, remove OT, 64 policy components,
64 candidates from their mixture, maximize marginal likelihood; T=.25;
five MuJoCo environments, seeds 0/1/2, 1M steps. Preserve DIPO.

Base: heejoon commit 4fb1d7d. Reuse its existing Direct GMM path under
analysis_tools/studies/20260918_nonstationary_nd/v5 unchanged. The top-level
legacy explorer is NOT the runtime for this experiment. No resampling,
Sinkhorn, assignment, row argmax, or conditional-row NLL is used.

For each state: z_i ~ N(0,I), i=1..64; obtain shared-network (mu_i,sigma_i).
q_old = (1/64) sum_i tanh-pushforward N(mu_i,diag(sigma_i^2)). Draw 64 iid
candidates by uniformly choosing components then drawing Gaussian noise.
Use the identical tanh-corrected mixture density for importance correction:
w = softmax(Q_mean/.25 - log q_old). Freeze candidates and weights.
Minimize -sum_j w_j log[(1/64) sum_i k_theta(b_j|s,z_i)], differentiating
both mu and sigma through logsumexp. The fixed-teacher tanh Jacobian may be
omitted from this optimization loss (but never from importance weights).

The reused proposal floor equals the actor's exp(-5) scale bound, so it does
not change any component scale. This is NOT an extra KDE fitted to sampled
actions, and NOT the teacher-only fixed sigma from v10.

Defaults: shared 256x2 GELU mu/std actor, scalar twin 256x2 critic, no LN or
gradient clipping, lr=3e-4, plain min-Q TD (no entropy), mean-Q extraction,
warmup5K, replay1M, batch256, UTD1. Collection/TD sample both z and epsilon;
dual evaluation uses stochastic-z/zero-z with epsilon=0, 10 episodes/5K.
Checkpoint50K. No external exploration noise or uniform replacement.
W&B: OptiQ/heejoon-direct-mll, fresh run IDs, no resume/early stopping.
Inherited OT fields are unused compatibility fields, NOT executed operations.

GPU allocation (15 non-DIPO slots out of 20 total):
- vast5 5090x4: Humanoid seeds0/1/2, Ant seed0
- vast1 4090 GPUs0/2/3: Ant seeds1/2, HalfCheetah seed0
- vast3 3090x4: HalfCheetah seeds1/2, Walker seeds0/1
- vast4 3090x4: Walker seed2, Hopper seeds0/1/2

Launchers verify committed source hashes, one run/GPU, isolated outputs,
actual GPU backend, saved config, and 1M checkpoints + dual evaluations.
Do not claim monotone policy improvement or exact Boltzmann recovery from
this finite-sample marginal-likelihood experiment.
