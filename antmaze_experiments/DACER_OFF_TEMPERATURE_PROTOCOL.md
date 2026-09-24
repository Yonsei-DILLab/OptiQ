# AntMaze OptiQ temperature-only comparison with DACER OFF

User approval on 2026-09-24: run only OptiQ v1/v3/v4 at fixed temperatures
3, 5, and 10, seed0 each (nine fresh policies). Keep replay1M to match the
baselines. This changes only teacher temperature relative to the completed
DACER-off T1 control, source484f92e7d6d34c964d85b4493ff17c5a9ebcf32e.
Do not adopt the discussed replay6M, clipping, learning-rate, sigma or critic
changes. No v2, extra seeds, baselines, annealing or cancelled-job restarts.

Preserve dense reward=-nearest-goal Euclidean distance, no sparse goal bonus,
NovelD OFF, DACER OFF, beta1, fresh normal latent and conditional sigma in
training/direct-policy evaluation. Actor/twin-critic256x3 GELU, mean-init1,
N=M64, log sigma[-5,-1]/initial-1, actorLR3e-4, criticLR5e-4, Adam without
gradient clipping, tau.005, gamma.99 and plain TD remain unchanged.

Keep256CPU environments, batch4096, eight learner updates per256transitions,
replay1M and8192warmup. Native post-warmup budgets are v1=3M, v3=4M, v4=5M;
the existing strict-greater/vector-rounding counter gives total transitions
3,008,256;4,008,448;5,008,384. Train from scratch after each job's own actual
8448transition/8update preflight, including full checkpoint/replay readback
and verified DACER-off behavior. Do not wait for all preflights or mazes.

Both four-5090 hosts receive18M native transitions. Host180 starts v4T3,
v3T5,v1T3,v1T5, then backfills v1T10. Host199 starts v4T5,v4T10,v3T3,v3T10.
Respect GPU locks and backfill local slots every2seconds. Preserve all existing
controllers, sources, results and cancellations. Failure holds pending jobs
while preserving live jobs; no automatic restarts. Commit/push/share before
preflight or training starts. W&B project OptiQ/antmaze, campaign/group
antmaze-optiq-dense-dacer-off-T3-T5-T10-s0-20260924.

Keep250k evaluation intervals and40 random-start episodes for each native and
direct-policy mode, plus evaluation-only intermediate policy checkpoints.
Final100episodes per native/direct/zero-z mode and random/fixed reset, with
full replay/model/optimizer/RNG/simulator state and SHA256 verification.
Training resets retain upstream defaults (random v1, fixed v3/v4); primary
evaluation uses xy uniform[-2,2] and records full initial states. No evaluation
intrinsic reward or extra behavior noise. Preserve raw trajectories and failures.

Compare against same-task T1 controls at equal steps. Report training visits,
left/right or upper/lower corridor use, route persistence across checkpoints,
same-state continuation and success separately. Action entropy alone does not
establish multiple complete routes. Each configuration has only one seed.
