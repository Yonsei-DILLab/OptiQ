# OptiQ Experiment Conventions

- This checkout is `/root/OptiQ-v5`, branch `v5`. Default entry `mujoco_v5`,
  alias `v5/final`; read `docs/v5/PSEUDOCODE.md` and `docs/v5/CHANGES_KO.md`.
  v5 sets `actor.ot_student_action=mean`: only OT student positions use
  tanh(mu). Preserve Gaussian teacher/density, full-row NLL for both heads,
  Gaussian collection/TD actions, and student epsilon RNG draws.
  Defaults: fixed T=.25, 256x2 actor/critic, initial sigma=.5, plain TD,
  no extra uniform replacement or annealing, and both epsilon=0 evaluations.
  Use `OptiQ/v4-test` with v5 groups, fresh IDs, and v5 output directories.
  Explicit v2/v3/v4 profiles retain sample-action OT when the new field is absent.
  The following sections describe preserved historical profiles.

- Optional user-requested v5 exploration profiles: `mujoco_v5_behavior010`,
  `mujoco_v5_annealing`, `mujoco_v5_annealing_behavior010`. Read
  `docs/v5/EXPLORATION.md`. They retain mean-action OT and plain TD, with
  p=.1 uniform collection and/or the existing post-warmup 10-to-.25 log-space
  40K temperature schedule. These are opt-in; canonical `mujoco_v5` stays
  fixed T=.25 with no added uniform replacement.

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

- Optional user-requested teacher annealing: `mujoco_v4_annealing`, alias
  `v4/annealing`; read `docs/v4/ANNEALING.md`. Log-space T=10 to .25 over
  20K or 40K environment steps after 5K warmup, then hold .25. Support p=0
  and p=.1 uniform collection; retain plain TD and OptiQ/v4-test. Canonical
  v3/v4 fixed-temperature profiles remain unchanged.

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
