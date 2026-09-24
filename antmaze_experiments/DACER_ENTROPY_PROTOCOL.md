# AntMaze OptiQ DACER entropy sweep

User-approved 2026-09-24. The user explicitly confirmed NEGATIVE per-action-dimension
targets: -1, -0.8, -0.5, -0.3, -0.1. For eight action dimensions, the regulator's
total targets are -8, -6.4, -4, -2.4, -0.8. These are targets for the existing
fitted-GMM entropy proxy, not a guarantee of exact policy or trajectory entropy.

## Scope and fixed settings

- 15 fresh policies: OptiQ only, v1/v3/v4 x five targets, seed 0.
- Constant teacher temperature 1; no temperature annealing.
- Dense negative next-position nearest-goal Euclidean distance; no sparse bonus.
- NovelD OFF; no RND model or updates. No reward scaling, shaping or soft TD change.
- Compare with completed T=1/DACER target -0.9 control, source
  7af193833466f5bc41853948d6f417fc6aaa6803. Only target_entropy_per_dim changes.
- 256 upstream Gym environments, batch 4096, eight updates per vector step,
  replay 1M, 8,192 warmup transitions. Actor/critics 256x3 GELU, Adam
  actor 3e-4 / critic 5e-4, tau .005, beta 1, random latent, N=M64,
  log sigma [-5,-1], initial -1, mean initialization scale 1.
- Keep DACER behavior-only, noise_scale .1, initial_alpha .27, alpha_lr .03,
  interval 10,000 learner updates, 3 fitted components, 200 samples per state,
  256 diagnostic states, entropy seed 42. First adaptation occurs at the first
  learner update; the interval and estimator are unchanged.
- Keep native per-maze budgets. Total transitions including warmup/vector
  rounding: v1 3,008,256, v3 4,008,448, v4 5,008,384.

## Evaluation and storage

- Every 250k transitions: 40 random-start episodes each for direct-policy and
  native mu-only evaluation. Save evaluation-only OptiQ policy checkpoints.
- Final: 100 episodes each for policy/native/zero_z, with random starts primary
  and fixed-full-state starts supplementary. Preserve every failed trajectory.
- Direct policy includes random z and conditional sigma; native removes sigma.
  Neither evaluation adds external DACER exploration noise or intrinsic reward.
- Final full model/optimizer/replay/RNG/simulator state with readback and SHA256.
- Keep the original training reset distributions; evaluation xy uniform[-2,2].
- Preserve exploration/entropy_proxy, exploration/target_entropy, alpha and
  noise_std in W&B, plus local progress and dacer_regulator.json.

## Scheduling and provenance

Campaign: antmaze-dense-dacer-entropy-T1-s0-20260924.
Project: OptiQ/antmaze; group equals campaign name.
Round-robin entries across vast-heechan-180 and vast-heechan-199 give 8 and 7 jobs.
Use existing GPU locks, one job per GPU, independent two-second backfill without
cross-maze/method barriers. Run each job's fresh 8,448-transition preflight first,
verifying actual batch4096 updates, signed DACER target applied by the regulator,
model/optimizer configuration, full checkpoint and dense replay rewards.
Failure holds pending entries while preserving live jobs; no automatic restart.
Controllers use the immutable committed source. Earlier campaigns and snapshots
remain unchanged; no additional methods, v2 or seeds are launched.

Both hosts currently have working W&B connectivity and use online logging.
The registered sync sidecar also supports explicitly selected offline logging
without changing training. It does not relaunch failed jobs.
