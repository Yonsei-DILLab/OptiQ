# Current AntMaze: dense reward, NovelD OFF, sixteen policies

User approval on 2026-09-24: run sixteen fresh dense-reward experiments without
NovelD. OptiQ/SAC/DIPO/MFPO x v1/v2/v3/v4, training seed0. This supersedes earlier
four-OptiQ-only and sparse/NovelD launch selections, without resuming cancelled
jobs or altering historical results/frozen sources.

## Learning configuration

- Official vendored DDiffPG environment files remain unchanged. The existing
  external wrapper returns `-min_goal ||next_xy - goal||_2`, with no sparse bonus,
  other shaping or normalization. Preserve physics, observations, action bounds,
  goal radius, termination and native training reset distribution.
- NovelD is disabled: zero intrinsic reward, zero RND updates, and no RND model
  or optimizer in saved checkpoints. SAC/DIPO retain the native compatibility
  dispatch label but record `enabled=false, effective_type=off`.
- All methods:256 Gym CPU environments, batch4096, replay1M, warmup8192, eight
  learner updates per256 transitions. Native budgets retain warmup and original
  strict-greater-than stop: v1/v2 3,008,256; v3 4,008,448; v4 5,008,384 total
  interactions, corresponding to93,752/125,008/156,256 learner updates.
- OptiQ retains the latest approved actor AND twin-critic256x3 GELU, actor LR
  3e-4/critic LR5e-4, Adam, tau.005, T=.01, beta1, DACER on, mean-init1, random
  latent,N=M64,log sigma[-5,-1],initial-1. Disabling NovelD also disables its RND
  estimator; DACER behavior exploration is a separate mechanism and stays on.
- SAC/DIPO/MFPO preserve their existing native architecture, learning rates,
  optimizers and objectives. Native DIPO uses5 diffusion steps,20 action-gradient
  steps, actor/critic LR3e-4/5e-4 and gradient norm1. Existing float64 C51
  projection/BCE/finite-gradient safeguards remain. Dense DIPO retains the
  already validated negative-support override[-6000,5],51 atoms. Its coarse
  spacing is an acknowledged compatibility limit, not an upstream default.

## Evaluation and saved state

Every250k:40 random-start episodes each for native and direct-policy modes.
OptiQ also saves its existing evaluation-only policy state every250k. Full
replay/learner/optimizer/RNG/simulator checkpoint remains final-only. Final
evaluation is100 episodes per mode/reset, including OptiQ zero-z separately.
Primary trajectories sample xy uniformly from[-2,2] for all mazes; identical
full-state rollouts are supplementary. This evaluation override does not alter
training resets. Direct sampling retains each policy's stochasticity, including
OptiQ conditional sigma. Native OptiQ uses random-z mu-only, SAC mean, MFPO
Q-best-of10 and DIPO native diffusion. No external DACER noise or intrinsic
reward is added during evaluation.

W&B uses OptiQ/antmaze with a new campaign group. Final100-episode metrics are
written under `final/<mode>-<reset>/...` and `final_evaluations`, separate from
periodic40-episode metrics. Frozen prior learning sources stay unchanged.

## Scheduling and validation

Campaign:antmaze-dense-off-16-current-s0-20260924. Host180 owns v1/v3, host199
v2/v4. Each host starts its two DIPO and two OptiQ runs first, then SAC and MFPO
independently fill each released GPU slot every2seconds. No maze/method barrier.
Existing GPU locks are respected; supervisor uses no automatic restart.

Commit/push and share identical frozen source before registration. Each job
runs one real256-env,batch4096,8-update preflight before main training. Verify
actor/critic changes and finite values, zero RND, complete checkpoint readback,
all replay rewards against next-state distances, evaluation reset behavior and
OptiQ's actual256x3 kernels/optimizer rates. No performance gate or hyperparameter
search is added. Failures hold pending jobs on the affected host and preserve
other live jobs. All older cancelled experiments remain inactive.
