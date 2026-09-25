# Matched random-start v3/v4 OptiQ, 2026-09-25

Stop the active v3/v4 T=3 basic Euclidean controllers and W&B sync services on
vast-heechan-180/199. Preserve their frozen source, logs, evaluations and
partial runs. Do not resume them. The N=M128 campaign on vast1 is independent
and continues unchanged; do not use vast1 for these new AntMaze jobs.

Train one fresh seed-0 OptiQ v3 on 180 and v4 on 199. Both training and primary
evaluation use the official DDiffPG Ant `random_init=True` reset: independently
sample XY uniformly over [-2,2] on every reset, with original pose/velocity.
This is the same random-start mechanism used for v1. It is an explicit change
from the native fixed v3/v4 start, so compare random-start runs separately from
the historical fixed-start runs. The final fixed-start supplement repeats one
sampled full state; primary 50k evaluations use fresh random starts.

The remaining basic OptiQ profile stays unchanged: N=M64, fresh Gaussian z,
T=1, beta=1, DACER and NovelD off, actor/twin critic 256x2, mean-head scale
1e-4, log sigma [-5,-1] initially -1, actor/critic Adam 3e-4, gamma .99,
tau .005, 256 parallel environments, batch 4096, 256 learner updates per 256
transitions, replay 1M and warmup 8192. Reward is nearest-goal Euclidean
`100*(d(current)-d(next))` with no step penalty or success bonus. Original
termination and v3/v4 horizons remain. Use native post-warmup budgets: v3 4M,
v4 5M, strictly stopping at the first 256-transition boundary above those
values. Every 50k transitions evaluate 40 episodes per native random-z mu-only
and direct sigma-inclusive stochastic policy; retain policy checkpoints. Final
evaluation has 100 episodes per mode/reset and the full checkpoint.

Commit/push the source and freeze it on both servers before preflight or main
training. The real preflight must verify 256 distinct training initial states
spread across the upstream XY range, nonidentical natural evaluation starts,
the exact model/reward/update configuration and full checkpoint readback.
Controllers use existing GPU locks, independent backfill and no automatic
restart. W&B project is OptiQ/antmaze. Report same-policy route diversity
separately from diversity caused by differing initial positions.
