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
