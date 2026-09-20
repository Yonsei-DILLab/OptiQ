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
