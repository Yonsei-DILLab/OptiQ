# OptiQ Experiment Conventions

- Latest user goal 2026-09-15: investigate failed best-k combinations and
  successfully integrate best-k while respecting existing OptiQ components;
  all four GPUs may be used. Read `docs/v5/BESTK_PROPOSAL.md`.
  `mujoco_v5_bestk_proposal` / `v5/bestk_proposal` uses independent best-of-8
  pilots to guide an explicit Gaussian proposal, retains original proposal mass,
  and applies full exp(Q/T)/q_mix correction. Preserve mean OT/full NLL/plain TD
  and existing evaluations. This first controlled variant keeps original
  Gaussian collection (Kb=1); do not label it collection best-k.
  It is under empirical investigation, not a demonstrated successful method.
  Random-pilot control `mujoco_v5_random_proposal` / `v5/random_proposal` keeps
  the same mixture, pilot/teacher counts and OptiQ components; selects the first
  IID pilot instead of argmax (`proposal_pilot_selection=first`). Its pool size
  is 8 but best-k selection is OFF. Use explicit random-pilot run labels.
  Full combination `mujoco_v5_bestk_combined` / `v5/bestk_combined` adds Kb8
  collection to the corrected best-k proposal. It keeps T/density/OT/NLL/Kt1
  and evaluation unchanged. Compare to proposal-only Kb1; success of Kb1 alone
  does not establish success of the originally requested collection combination.
  Latest user steering on 2026-09-15 permits rational performance early stops:
  continue promising/recovering runs up to 1M, stop persistently poor runs and
  explore alternatives. Assess both evaluation modes, matched meanOT windows
  and recovery trends; record the evidence before stopping. This overrides
  the older default prohibition below for the current best-k investigation.
  Compare fresh runs against matched meanOT seeds; report partial budgets
  honestly and never mark early-stopped runs as completed 1M. Keep historical
  winner-only/collection-only source snapshots and logs intact.
  Work log: `/root/anal/optiq_bestk_integration_20260915/WORKLOG.md`.

- Experimental worktree `/root/OptiQ-v5-bestof8`, branch `v5_bestk`:
  user requested best-of-8 collection AND winner-distribution OT distillation.
  Read `docs/v5/BEST_OF_8.md`. Opt-in `mujoco_v5_bestof8` / `v5/bestof8`
  uses Kb=8, teacher K=8, Kt=1: 64 independent full-Gaussian winners,
  uniform teacher mass, no Boltzmann temperature or density correction.
  Keep mean-action student OT/full conditional NLL and both evaluations unchanged.
  W&B `OptiQ/v5-bestk`; fresh 1M seed0 runs then queued seed1 after both finish.
  Preserve cancelled collection-only snapshots. Seed4 recovery remains on hold.
  Canonical v5 and its historical profiles remain unchanged.

- This checkout is `/root/OptiQ-v5`, branch `v5`. Default entry `mujoco_v5`,
  alias `v5/final`; read `docs/v5/PSEUDOCODE.md` and `docs/v5/CHANGES_KO.md`.
  v5 sets `actor.ot_student_action=mean`: only OT student positions use
  tanh(mu). Preserve Gaussian teacher/density, full-row NLL for both heads,
  Gaussian collection/TD actions, and student epsilon RNG draws.
  Defaults: fixed T=.25, Sinkhorn epsilon=.1 / 100 iterations, no actor or
  critic gradient clipping (ac_grad_norm=null), 256x2 actor/critic, sigma=.5, plain TD,
  no extra uniform replacement or annealing, and both epsilon=0 evaluations.
  Current Ant grid uses `OptiQ/v5-jaehoon`; pass `wandb.project=v5-jaehoon`
  explicitly because YAML retains the historical v4-test destination. Older
  meanOT/exploration records remain in v5-test. Read docs/v5/EXPERIMENTS_KO.md.
  The registered six-cell epsilon/T grid uses frozen source 71c5ba8 with explicit
  overrides. Do not edit its source, manifest or workers when updating this repo.
  Explicit v2/v3/v4 profiles retain sample-action OT when the new field is absent.
  The following sections describe preserved historical profiles.

- Optional user-requested v5 exploration profiles: `mujoco_v5_behavior010`,
  `mujoco_v5_annealing`, `mujoco_v5_annealing_behavior010`. Read
  `docs/v5/EXPLORATION.md`. They retain mean-action OT and plain TD, with
  p=.1 uniform collection and/or the existing post-warmup 10-to-.25 log-space
  40K temperature schedule. These are opt-in; canonical `mujoco_v5` stays
  fixed T=.25 with no added uniform replacement.
  Optional v5 profiles inherit the current epsilon=.1 / no-clip defaults;
  historical exploration runs used epsilon=.25 / clipping=2.0. Use explicit
  overrides to reproduce those historical settings.

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
