# 26 September: finish by 16:00 KST, preserve all previous attempts

Latest user instruction replaces the running/pending long PointMaze,
Medium sensitivity, MEOW alpha and previous Way queues. Stop their learners
and controllers, preserve data, and never automatically resume them.
Use only the healthy 199 server (4 RTX 5090) and new 46 server (8 RTX 5090).

Requested results: wall-free 4/8/16-Way OptiQ T=1,3,5,10 (12 policies), and
Simple/Medium/Hard PointMaze x OptiQ/SAC/SQL/MEOW/MFPO/DIPO/TD3 (21 policies).
All are seed 0. Existing verified 8/16-Way T1 results at exactly 1M are reused,
as explicitly requested. They used 16env, batch256, UTD1, 998976 updates;
label them historical and do not present the temperature grid as matched.
Other saved long PointMaze evaluations were at 400k/600k multiples, not 1M;
do not relabel a later checkpoint or concatenate interrupted runs.

The 31 fresh jobs use the established fast profile: 256env, batch4096,
16 updates per 256 transitions, warmup8192, 1,000,192 total transitions
(first complete vector batch at 1M), exactly62000 learner update calls.
This does not equal the historical UTD1 optimization budget. Native models,
optimizers, rewards and policy evaluation remain unchanged. PointMaze OptiQ
retains its actual previous T3; wall-free OptiQ uses the requested T grid.
OptiQ DACER stays OFF, random z, N=M64, 256x2, log sigma[-5,-1]. Baseline
settings remain in POINTMAZE_LONG_BASELINE_AUDIT.md, without the cancelled
Medium hyperparameter sweep. SQL T1/16 particles, MEOW alpha.2, MFPO -.5.

Evaluation every~200k:200 episodes, final PointMaze500 and Way1024.
This returns PointMaze final evaluation to the original 500-episode protocol
instead of the cancelled long run's2000 to meet the deadline. Keep native
direct-policy rollouts, separately labeled OptiQ random-z mu-only, obstacle
and goal-removal statistics, goal counts including failures, Q/action probes,
automatic figures, policy/critic checkpoint and final replay. No sigma/mu
relabeling. Step/update/config/raw goal counts and SHA256 must pass before
a job is complete. Run a fresh8448-step/16-update preflight before each job.

Commit merged exact code and this plan, push and export immutable source
including pinned MFPO/MEOW submodules before starting. Copy the established
199 runtime to46, verify ownership, CUDA/JAX/Torch and real updates there.
199 prioritizes all3 DIPO jobs then other PyTorch baselines.46 schedules
OptiQ/SQL/MFPO. Each host uses one shared locked queue with independent GPU
backfill and no maze barrier. Respect existing GPU locks and compute PIDs.
No automatic restart or silent parameter/budget reduction on failure.
The 16:00 deadline is a target, not fabricated completion evidence: report
any remaining work honestly. Store reused and fresh provenance separately.

## Additional host approved during setup

User added vast-heechan-6 (136.63.24.6:41636), four RTX5090. Total16 GPUs.
Move only the six unstarted PointMaze OptiQ/SQL jobs from46 to6; retain13
jobs on46 and the already-running/pending12 on199. Preserve frozen training
source a55aaf13f7803b5cf8da7ba56f318675c7794171 on every host. The later
controller/report commit changes placement only, not learner code. New
controllers run from their own immutable source and pass the original frozen
training path to every preflight/main subprocess. Record both paths in manifests.

## Pending-only redistribution after actual throughput check

To use the new hosts when their first jobs finish, release all eight unclaimed
199 jobs under queue.lock: six SAC/TD3 jobs move to6, Medium/Simple MEOW to46.
Never move an already claimed job. Commit the placement and transfer tool first.
Write a durable source release before destination acceptance; retain original
manifests and source entries as transferred, with audit sidecars. Existing
workers claim appended jobs normally, retaining original a55 frozen training.
No optimizer/batch/update/temperature/evaluation change or learner restart.
After owned jobs complete, finalize the drained source queue while preserving
its transfer provenance. Global completion still requires all33 unique results.
