# Wall-free 4/8/12/16-Way 1M grid (2026-09-26)

Run seed 0 for OptiQ at teacher temperatures 1, 3, and 5 plus six baseline
methods (SAC, SVGD SQL, MEOW, MFPO, DIPO, TD3) on each task. This is 36
policy/task combinations. Reuse the three completed 8/12/16-Way OptiQ T=1
1M runs from source `973eca4a432cf027a5c725a372aafb86b9ed6c83` after
their source/config/checkpoint/rollout SHA verification. The remaining 33
jobs use this committed source; do not relaunch those three controls.

The official wall-free 4-Way task retains its four goals at cardinal radius
5, horizon 20, and its existing reward/termination. The 8/12/16-Way ring
tasks share radius 6, success radius .5, horizon 24, and their existing
reward. All policies start at the center. Task geometry differs between
4-Way and N-Way; compare methods within each task, not absolute returns
across tasks.

All runs use 16 vector environments, batch 256, 16 learner updates per
16 newly collected transitions (UTD 1), warmup 1024, exactly 1M collected
transitions and 998,976 learner updates. These are the established 4-Way and
N-Way settings and keep the existing T=1 controls comparable. Increasing
environment count without changing the update count would reduce UTD;
matching UTD instead leaves the same 998,976 optimizer updates, the dominant
work for expensive baselines. Baselines retain their native architecture,
optimizer, and policy sampling; only OptiQ teacher temperature varies.

Each job performs a fresh 1040-transition/16-update preflight. The main run
evaluates its directly sampled policy every 100k transitions for 256 episodes
and at 1M for 1024 episodes. OptiQ also stores the separately labeled
random-z mu-only evaluation. Save raw trajectories/goal IDs, policy/Q probes,
per-evaluation trajectory figure, policy/critic checkpoints and final replay.
The 4-Way figure uses the shared DACER-style reference contour plus sampled
policy arrows and the learned critic surface; the contour is not policy
density. Success rate, goal counts, and per-goal coverage are all reported.

Use guarded, independently advancing GPU shards with no all-shard barrier.
GPU2 on vast-heechan-180 is reserved for the nine 4-Way jobs. The three
existing N-Way baseline guards on 180 GPU0/3 and 199 GPU3 are unclaimed;
stop those guards and transfer their pending 18 jobs to this source before
registration. Record the cancellation/transfer sidecar and preserve their
original plans and logs. Those three GPUs are currently used by PointMaze;
new guards wait for the existing locks and actual idle hardware. Each N-Way
shard adds OptiQ T=3 and T=5, then its six baselines. Never launch a second
copy from the old queue. A failed job stops only its shard and preserves
all completed/partial data; no automatic retry or altered hyperparameters.

The grid is descriptive: one training seed per method/temperature/task.
Wall-free mode coverage is not established by aggregate success alone.
