# Replacement for the interrupted random-start 100Δd controls

The random-start v3/v4 OptiQ `100*(d(current)-d(next))` controls were stopped
in error at 147,456 and 122,880 transitions, respectively. They were not
declared failed. Their only intermediate saves contain evaluation policy state,
not full replay, optimizer, simulator and RNG state, so exact continuation is
impossible. Preserve both partial campaigns and start two fresh seed-0 controls
with the same learning/evaluation settings and native budgets.

Use the committed `register_utd256_random_starts` manifest unchanged except
campaign/job identifiers and provenance. This means v3/v4 upstream random
XY[-2,2] starts for both training and primary evaluation; T1, N=M64, OptiQ
basic 256×2, mean-head scale1e-4, log sigma[-5,-1] initially -1, DACER/NovelD
off, `100Δd` nearest-goal Euclidean reward with zero success bonus and zero step
penalty. Keep 256 parallel environments, batch4096, 256 updates/256 new
transitions, replay1M, warmup8192, seed0, and native 4M/5M post-warmup
budgets. Evaluate every50k with 40episodes/mode; retain evaluation-only policy
checkpoints and final full state. W&B is OptiQ/antmaze.

Run v3 on vast-heechan-180 and v4 on vast-heechan-199. Their two-reward
campaign already uses GPU0–1 on each host; use the next eligible unlocked GPU.
Commit/push/freeze the source before launch. Require the usual real 8,448-step,
256-update preflight and verify reward, starts, N=M64 and full preflight state.
No automatic restart, and do not touch the cancelled T3/N128 experiments.
