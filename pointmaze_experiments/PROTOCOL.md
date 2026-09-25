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
