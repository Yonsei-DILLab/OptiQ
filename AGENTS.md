# OptiQ Experiment Conventions

- On branch `v4`, the default entry is `mujoco_v4`, alias `v4/final`.
  Read `docs/v4/PSEUDOCODE.md`. Actor and critic default to 256x2. Evaluate
  both zero-z and stochastic-z with epsilon=0, separately logged, using paired
  episode reset seeds and isolated evaluation RNG. Keep v3 training semantics.
  Log v4 to `OptiQ/v4_test` unless the user overrides it. The instructions below
  describe the preserved historical v3 baseline.

- Optional 10% uniform collection: `mujoco_v4_behavior010`, alias
  `v4/behavior010`; read `docs/v4/BEHAVIOR010.md`. It uses the existing
  collection hook after warmup and logs to user-requested `OptiQ/v4-test`.
  Default v4 remains p=0. Enabling the profile does not imply permission to
  replace or restart an existing frozen experiment campaign.

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
