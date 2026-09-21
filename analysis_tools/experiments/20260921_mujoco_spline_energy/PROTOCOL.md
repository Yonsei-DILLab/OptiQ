# State-conditioned Spline Energy Circuit on MuJoCo

## Question

User request, 2026-09-21: run the spline energy model on Hopper, Walker2d,
HalfCheetah, Ant and Humanoid with three training seeds (15 runs total).

Hypothesis: a state-conditioned positive spline circuit can tie policy and Q in
one model, retain one-forward exact policy sampling, and learn useful MuJoCo
control policies without a separate actor. This is the first online RL test of
the GMM40 idea, not an architecture-only comparison or a claim of superiority.

## Model and deliberate changes from the GMM40 toy

For each state, one 256x2 LayerNorm MLP emits a scalar V, root logits, and
piecewise-linear leaf log-heights:

    pi(a|s) = sum_r softmax(h(s))_r product_d f_rd(a_d|s)
    Q(s,a) = V(s) + alpha log pi(a|s)
    integral exp(Q(s,a)/alpha) da = exp(V(s)/alpha)

Every leaf is normalized by its exact trapezoid integral on [-1,1]. Sampling
chooses one root and analytically inverts every linear-density leaf CDF in
parallel. There is one neural forward and no Gaussian sigma, tanh compression,
diffusion, MCMC, action search, or separately trained actor/critic. An EMA copy
of the same circuit supplies the TD target; it is not a second model family.

GMM40 used rank64 and129 knots in two action dimensions. The MuJoCo version uses
rank16 and33 knots so Humanoid's17-dimensional output remains practical. This
is a documented scaling change. Target-independent initial leaves are broad
bumps centered at fixed random points in [-.5,.5]^D. State-dependent head
kernels break symmetry. Temperature alpha=.25.

Fit the soft Bellman equation with Huber TD loss:

    y = r + .99 (1-terminal) V_target(s')
    L = mean huber(Q(s,a)-y), delta=10

Use Adam3e-4, global gradient norm cap10, target tau=.005, batch256, one update
per environment step after5,000 random warmup steps, replay capacity1,000,000.
Time-limit truncations bootstrap; true terminations do not. No reward scaling,
observation normalization, prioritized replay, or post-hoc action selection.
The circuit and replay use normalized actions in [-1,1]; an affine map applies
each environment's physical Box bounds at `env.step` (notably Humanoid is
[-.4,.4]).

## Runs and evaluation

Gymnasium v4 tasks: Hopper, Walker2d, HalfCheetah, Ant, Humanoid. Seeds0,1,2;
1,000,000 environment steps each. Evaluate the native stochastic circuit for10
episodes at step0 and every10,000 steps, with fixed reset seeds and evaluation
randomness isolated from collection. Primary metric is `eval/mean_reward`; log
standard deviation, episode length, return AUC, TD loss, Q, V and gradient norm.

W&B entity/project/group:
`gsmin2018/OptiQ-MuJoCo-Spline-Energy/spline_energy`. The config field
`algorithm=spline_energy` supports algorithm grouping; `algorithm_display_name`
keeps the human-readable `Spline Energy Circuit` label. Preserve final
and100k-interval checkpoints, configs, evaluation JSONL, raw logs, source hash,
environment/package/GPU metadata and SLURM IDs.

Before the 15-run launch, validate exact normalization/sampling and a nonzero
finite TD update on CPU. Because state conditioning, TD gradients and environment
interaction are new, run one committed GPU smoke covering Hopper and Humanoid
with real5k replay warmup and several hundred updates. Submit the full array only
after it passes. Main jobs use one GPU/four CPUs/32GB, big_qos,72h; previous
HalfCheetah jobs are not modified.
