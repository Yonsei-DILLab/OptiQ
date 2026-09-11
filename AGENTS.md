# OptiQ Experiment Conventions

- This checkout is `/root/OptiQ-v3`, branch `v3`. The original baseline
  remains `/root/OptiQ`, branch `v2`. Build v3 from continuous-latent
  checked-K64; do not substitute the historical Gaussian-W2 variant.
- Canonical v3 config: `configs/v3/final.yaml`, alias `mujoco_v3`.
  Read `docs/v3/PSEUDOCODE.md` before changing or launching v3.
- v3 uses plain TD, no policy entropy evaluation/bonus, no soft guard
  or optimizer rollback. Keep conditional-mixture teacher, full OT conditional
  NLL, teacher T=.1, OT epsilon=.25/100 iterations, beta=1, gradient clipping=2,
  LayerNorm=false, and no uniform collection or annealing.
- Log comparable v3 runs to `OptiQ/optiq_mujoco_v2_confirmation`.
  Other experiment families retain their own existing baseline project.
  Distinguish variants with explicit run names, groups and configuration fields.
- Fresh run IDs and output directories only. Never overwrite or resume an old
  run as a new comparison. Preserve historical code/config/result snapshots.
- Default v3 outputs: `/root/optiq-experiments/v3_td/outputs`.
  Experiment records live outside the repo in `/root/optiq-experiments`;
  analysis lives in `/root/anal`. Use supervisor for long-running GPU workers.
- Run requested training to 1M environment steps per seed by default. No
  performance-based early stopping, score-collapse stopping, or inherited
  futility gates unless the user explicitly requests a different rule.
  Short tests must be labeled validation, not completed training experiments.
- Do not push unless the user explicitly asks.
- The pre-unification local edits are preserved in
  `/root/optiq-archives/v2-before-checked64-20260911` and a named Git stash.
  Do not reapply that stash onto canonical v2 as a routine setup step.
