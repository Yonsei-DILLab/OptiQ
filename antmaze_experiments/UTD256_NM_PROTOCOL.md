# OptiQ AntMaze N=M comparison, 2026-09-25

Fresh seed-0 v3/v4 Direct GMM/TRG policies compare N=M128 and N=M256 at
1,000,192 total environment transitions (the first 256-transition boundary
at or above the requested 1M; 8,192 warmup transitions are included). Each
run uses 256 parallel official DDiffPG environments, batch 4,096 and 256
learner updates for every 256 collected transitions. Exactly 992,000 actor
and critic updates are expected. N denotes policy latent samples per state;
M denotes teacher candidates per state. `proposals_per_policy_sample=1`, so
both equal the selected 128 or 256. The latent prior remains fresh Gaussian,
not a fixed codebook.

Only N/M and the requested 1M budget differ from the stopped T=1 basic
Euclidean control. All runs use `100*(d(current)-d(next))` with nearest-goal
Euclidean distance, no step cost or success bonus; OptiQ basic 256x2 actor
and twin critics, mean-head scale 1e-4, actor/critic Adam 3e-4, T=1, beta=1,
DACER and NovelD off, log sigma [-5,-1] initially -1, gamma .99, tau .005,
replay 1M. v3/v4 train and evaluate from the official fixed full state.
Every 50k transitions (aligned to 256), evaluate 40 episodes each with
random-z mu-only and direct-policy conditional-sigma actions; save evaluation
policies. Final evaluation uses 100 episodes per mode and full state save.

Placement: v3 N256 on vast-heechan-180, v4 N256 on vast-heechan-199,
v3 and v4 N128 on vast1. Controllers choose idle GPUs, respecting locks.
Current T=3 v3/v4 jobs continue on the two 5090 hosts. Prior 4090 Hopper
and NM GPU workers are stopped at the user's instruction; their artifacts
remain intact. No other queued work is resumed or started.

Commit and push this source before registration. Freeze the same full SHA
under `/home/heechan/OptiQ-ops/sources/<SHA>` on every participating host.
The existing real preflight performs one 256-transition collection and 256
batch-4096 learner updates after warmup. Its initial and final proofs must
show the requested policy sample count, one proposal per sample, and exact
actor/critic update counts. An OOM or nonfinite failure holds remaining work;
do not silently shrink N/M, batch, or UTD. Preserve logs, checkpoints,
source, and the stopped 4090 workloads.
