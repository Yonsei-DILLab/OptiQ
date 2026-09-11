# OptiQ Experiment Conventions

- Canonical repository: `/root/OptiQ`, branch `v2`. The default v2 is the
  successful continuous-latent checked-K64 algorithm, training commit
  `8cb4f237aacb61113f65e693e23236f17d77f978`, NOT the local Gaussian-W2 variant.
- Canonical config: `configs/v2/final.yaml`; `mujoco_v2` and
  `mujoco_v2_checked` are aliases. Read `docs/v2/PSEUDOCODE.md` and
  `docs/v2/INSTANCE.md` before launching or changing v2.
- Keep successful numerical defaults: conditional-mixture teacher, full OT
  conditional NLL, sampled soft guard with optimizer rollback, T=.1,
  gradient clipping=2, LayerNorm=false, no uniform collection or annealing.
- Log comparable checked-v2 runs to `OptiQ/optiq_mujoco_v2_confirmation`.
  Other experiment families retain their own existing baseline project.
  Distinguish variants with explicit run names, groups and configuration fields.
- Fresh run IDs and output directories only. Never overwrite or resume an old
  run as a new comparison. Preserve historical code/config/result snapshots.
- Default checked-v2 outputs: `/root/optiq-experiments/v2_checked64/outputs`.
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
