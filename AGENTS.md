# OptiQ experiment versioning

The user requested on 2026-09-17 that this work use branch `heejoon` and that
experiments be committed before they are launched.

- Commit the exact experiment code, configurations, launch scripts, and protocol
  before new training or a changed setting is launched. Record the full commit
  SHA with the source manifest and submitted jobs.
- An unchanged checkpoint resume retains its original source commit; document
  the resume without creating empty commits per seed or scheduler retry.
- Preserve running/frozen source snapshots. Keep Git provenance in a sidecar
  when adding metadata to already-running experiments, and disclose post-launch
  commits explicitly.
- Do not commit credentials, SSH/W&B keys, checkpoint/sample archives, virtual
  environments, or generated logs. Data remains on dildata. Small existing
  source-manifest fixtures may be retained for reproducibility.
- Do not force-push or overwrite collaborators' changes. Preserve shared branch
  history and the source/configuration used by earlier experiments.

Recent self-contained experiments are under `analysis_tools/studies/`.

User clarification (2026-09-18): MuJoCo environment order is a scheduling
preference, not a completion dependency. Independent runs should be eligible
together and use available GPUs concurrently. Use priority/nice to express
environment preference; add cross-environment `afterok` only when the user
explicitly requests a completion gate. Keep explicitly deferred runs deferred.

User default (2026-09-21): For new Direct GMM/TRG runs, keep actor log_std
bounds fixed at [-5, -1] (log_std_min=-5.0, log_std_max=-1.0). Explicitly
requested ablations may override these bounds in their own experiment profiles;
do not promote ablation bounds to defaults based on their results. Preserve
running/frozen snapshots. The requested Ant/Humanoid 20k cap 0 versus -2
comparison is an explicit experiment-only exception.

User visualization default (2026-09-21): OptiQ GMM figures use μ-only action
outputs with conditional Gaussian noise removed, retaining each run's original
fixed/random latent prior. Near/coverage/MMD labels, tables and curves must use
the same μ-only samples. Keep full-policy σ-noise results as a clearly labeled
supplement, never relabel full-policy metrics as μ-only. Other baselines retain
their native generator outputs and are labeled accordingly. Preserve raw metrics,
training, RL reward evaluation and frozen source. For old frozen campaigns,
render a separate post-hoc report with reporting-source provenance.

User constraint (2026-09-21, latest): Keep GMM40 OptiQ N=M=64 in this
parameter search. Do not increase N/M for the coverage/near goal. Existing
N=M256 results are historical only and do not satisfy the current goal.

User experiment selection (2026-09-22): Future AntMaze comparisons use OptiQ, SAC, DIPO, and MFPO; MEOW runs/queues were cancelled. The requested fresh NovelD10 v1-v4 seed0 1M campaign should evaluate every250k and save the full checkpoint only at final1M. The latest instruction holds new training until saved-checkpoint trajectories have been inspected to assess coefficient10. Preserve cancelled-run logs/checkpoints and do not relaunch older campaigns.
