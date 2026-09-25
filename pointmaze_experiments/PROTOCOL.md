# DrAC PointMaze pilot, 2026-09-26

User replaces D3IL with DrAC multi-goal PointMaze; stop non-MaxEntDP experiments.
Only Simple seed0 temperature1 is launched initially. No reward learning/experts.

- Upstream PneuC/DrAC commit 4e718983ea29aa3a856955f553a99795fcb4e94d;
  unchanged MultiGoalPointMaze, sparse goal+100/otherwise0, center reset, four
  goals, max150 steps, no test-only obstacles. Goal-free 4D state, 2D action.
- 100K total interactions including10K uniform warmup; one environment,
  one independently sampled batch256 learner update per subsequent transition.
- Unchanged Direct GMM/TRG iBOLT N=M64,beta1,256x2 actor and scalar twin critics,
  log_std[-5,-1]/initial-1, LR3e-4, gamma.99, ordinary TD, no DACER/NovelD/OT.
- Every5K:100 fixed-origin episodes EACH random-z mean-only and full policy,
  success, goal counts, entropy, trajectories PNG/NPZ and verified policy snapshot.
- One RTX5090 on vast2 GPU0; W&B OptiQ/jaehun-drac-pointmaze.
- Short preflight256 warmup+16updates, two evaluation episodes each mode,
  validates realGPU, reward, transition accounting and checkpoint roundtrip.
- No environment/reward changes, no borrowing DrAC diversity objective.
  One seed is a feasibility pilot, not a paper-level comparison.

Historical services disabled individually (autostart/autorestart false);
logs/checkpoints remain. MaxEntDP learners and infrastructure/log sync retained.

## Approved three-maze extension
User requests all three mazes and test-only light-gray obstacles. Preserve
completed Simple100K. Add Medium200K/Hard300K total steps, seed0 T1, unchanged
10K warmup and every learning setting. Native episode limits300/600 and4/8 goals.
Separate frozen suite checkout; Simple source/checkpoint/results are untouched.
Use vast2 RTX5090 GPU1/2 for Medium/Hard and GPU0 for Simple checkpoint evaluation.
Every5K retains100episodes per mode in the unmodified training maze.
Final robustness adds official maze_eval_mode=True obstacles, without learning,
500episodes per mode grouped into100 blocks of5 for success-within-five trials;
also retain single-episode success, goal counts and PNG/NPZ paths.
This matches the official trial-group metric but uses serial rather than16-vector
evaluation and native named-map episode limits. It is not an exact replication
of upstream's list-map constructor (which defaults to500steps).
No goal-removal experiment, extra seeds, temperatures or baseline launches.
Run real 16-update preflights with geometry checks and obstacle evaluation first.

## Approved temperature extension
Latest user explicitly adds T3 and T5 for all Simple/Medium/Hard, seed0: six
fresh jobs,100K/200K/300K including10K warmup. No learning/evaluation change
other than temperature; retain original sparse reward and final obstacle tests.
Preserve running T1 Medium/Hard on vast2 GPUs1/2 and MaxEntDP on other hosts.
Two independent serial queues on idle vast2 RTX5090 GPU0(T3)/GPU3(T5), each
Simple then Medium then Hard. Failures preserve results and do not block the
next independent maze; no retries or checkpoint resumes. Exact source frozen
in a separate worktree. Preflight both new temperatures before main launch.
