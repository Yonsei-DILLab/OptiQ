# Fast, internally matched 4/8/12/16-Way grid (2026-09-26)

This protocol supersedes the pending `WAY_GRID_1M_PROTOCOL.md` queue. That
queue never claimed a job on vast-heechan-199; preserve its frozen source,
plans and stopped supervisor logs. The first 4-Way preflight on
vast-heechan-180 failed before any learner update because new CUDA contexts
cannot initialize on that host. Preserve the failed preflight and all
existing PointMaze learners; do not reset/restart the instance.

Run all 36 method/task/temperature combinations afresh with one matched
profile. For each wall-free 4/8/12/16-Way task and seed 0, run OptiQ at
teacher temperatures 1, 3 and 5 plus native SAC, SVGD SQL, MEOW, MFPO,
DIPO and TD3. Prior 8/12/16-Way OptiQ T=1 results used 16 environments,
batch 256 and UTD 1; keep them as supplementary historical controls, not
points in this matched grid.

Use 256 vector environments, batch 4096, 16 learner updates after each
256 newly collected transitions, warmup 8192, and 1,000,192 total
transitions (the first multiple of 256 above 1M). This gives exactly
62,000 learner updates after warmup and UTD 0.0625. The profile matches
the established fast PointMaze data/update cadence. It is an explicit
speed/optimization-budget tradeoff, not the same learner-update budget as
the historical 16-environment runs. Every method/temperature in this new
grid uses the same profile. Baselines retain their native architectures,
optimizers and direct-policy sampling; OptiQ alone varies teacher T.

Each job passes an 8448-transition/16-update fresh preflight, then starts
a fresh main run. Evaluate every ~200k transitions (near the vector batch
boundary) with 256 directly sampled policy episodes; final evaluation
has 1024 episodes. OptiQ also stores separately labeled random-z mu-only
rollouts. Store goal IDs, full trajectories, action/Q probes, policy and
critic checkpoint, final replay and automatic paper-style figures. Do not
infer complete mode coverage from aggregate success. The 4-Way geometry
and reward differ from the 8/12/16-Way ring family; compare methods
within a task. One seed gives descriptive, not inferential, comparisons.

Run four independent guarded GPU shards on vast-heechan-199, one maze per
GPU, in a separate campaign root. Existing PointMaze and MEOW-alpha
workers keep their GPU locks; each guard waits for real zero compute PIDs
and the common flock, then its jobs advance without a global barrier.
No automatic retry, duplicate launch, or on-the-fly hyperparameter change.
The exact source, commands and job order are committed in the fast plan
and supervisor configuration. Failure stops the affected shard and
preserves completed/partial files for inspection.
