# Wall-free N-Way T=1 baseline comparison (2026-09-26)

Compare the existing OptiQ T=1 seed-0, 1M-transition runs on the symmetric
8/12/16-Way tasks against SAC, SVGD SQL, MEOW, MFPO, DIPO, and TD3. The
existing 4-Way comparison uses 100k transitions; its OptiQ panel must come
from the completed T=1 run, not the separate T=5 temperature selection.
The 32-Way OptiQ result remains an OptiQ-only expansion and is outside this
baseline grid.

The training source is the already frozen commit
`973eca4a432cf027a5c725a372aafb86b9ed6c83`. Do not modify it or any
existing run. This source provides all seven methods, the N-Way environment,
per-job 1040-transition/16-update preflight, exact 1M training, and raw
evaluations. Each new job uses seed 0, 16 vector environments, batch 256,
16 learner updates per 16 collected transitions (UTD 1), 1024 warmup
transitions, and the baseline's native architecture/optimizer. OptiQ alone
uses teacher temperature 1. All methods get exactly 998,976 updates after
warmup. The reward, ring geometry, success radius, and horizon are identical
to the existing OptiQ N-Way runs. Each method is evaluated by directly
sampling its policy; final success and goal counts come from 1024 episodes
from the fixed central start. Intermediate evaluations use 256 episodes
every 100k transitions. Store raw trajectories, Q/action probe, policy and
critic checkpoints, and final replay.

Three independent guarded GPU shards run six methods each. GPU 0/3 on
vast-heechan-180 and GPU 3 on vast-heechan-199 are currently occupied by
the earlier long PointMaze campaign, so each new supervisor waits for both
the lock and truly idle hardware before claiming a N-Way job. Preserve all
unrelated compute. No retry, restart, extra seed, or hyperparameter
adjustment is implicit. A failed job stops its shard and preserves pending
jobs. No all-shard completion barrier; each shard runs its next job
immediately. The exact job order and hosts are in
`NWAY_BASELINES_T1_PLAN.json`; source, scheduling commit, config and
supervisor status must be recorded with the campaign.

The main comparison is sampled-policy success, goal counts, and trajectory
diversity, not success alone. A single visit to a goal is shown separately
from material coverage (at least 1% of final evaluation episodes). Each
method has only one training seed, so this is descriptive rather than a
significance claim. MEOW's 4-Way T=1 output was strongly action-saturated;
keep its native parameters for baseline fidelity and report this caveat,
without silently tuning it.
