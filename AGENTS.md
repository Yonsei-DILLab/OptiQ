# OptiQ v8 current instructions

This checkout is /root/OptiQ-v8, branch v8. User-authorized v8 supersedes historical v7 algorithm instructions below. Implement fresh per-state Gaussian-likelihood OT plus conditional SAC and validate on GMM40 before RL campaigns.
- Actual actor Gaussians at H=4096 normal latent integration points define C=-alpha log pi_old,i(a|s). Entropic coefficient alpha (dimensionless likelihood epsilon=1).
- Teacher candidates M=256, one sample per fresh latent Gaussian; W=softmax(Q/alpha-log q), resample K=16 BEFORE OT. Preserve proposal-density and tanh/action-scale corrections. Proposal sigma floor never enters source Gaussian cost or actor density.
- Solve each state independently from scratch; no persistent dual model/optimizer. Balanced source 1/H, target 1/K; check row and column residuals (min/max 10/2000, relative tolerance 1e-3). Do not silently accept nonconvergence.
- Sample one source per teacher column of P. Actor uses newly generated actions and conditional SAC loss alpha log pi_i-Q-alpha log Pr_OT(i|a). No extra source importance ratio. Freeze old source Gaussian parameters and solved potentials, retain query-action derivatives.
- All H source Gaussian frozen forwards are required; actor parameter backward for selected K only. Reuse identical-state forwards in GMM only when exactly equivalent.
- Preserve plain MLP 256x2, learning rate 3e-4, sigma bounds/init, Q target, prior, evaluation protocol. GMM alpha=1, no unrequested learning-rate/UTD changes.
- Core JAX implementation should be reusable for RL; soft TD uses mixture policy density. Preserve historical v7 paths and external baseline original implementations.
- Store run artifacts outside repo; long GPU runs use supervisor and frozen source. User authorizes GMM experiments. Do not push without explicit instruction.

# Historical instructions (v8 overrides above take precedence)

# OptiQ Experiment Conventions

- This checkout is `/root/OptiQ-v7`, branch `v7`, based on v5 `a5d5e28`.
  Default entry is `mujoco_v7`; explicit historical profiles remain available.
  Read `docs/v7/ALGORITHM_KO.md`, `docs/v7/PSEUDOCODE.md`, and
  `docs/v7/NOTATION_KO.md`. Keep original teacher b_j, normalized W_j,
  coupling P_ij and row conditional R_ij notation in documentation.
  At the user's request the canonical objective is restored to
  `ot_conditional_sac`: source importance times
  `[T log pi(a|s,z_i) - Q(s,a) - T log Pr(i|a,s)]`.
  The goal is OT-assigned conditional Boltzmann extraction, not equality to
  marginal SAC. Explain population source balance and conditional fitting as
  sufficient recovery conditions, without claiming neural SGD convergence.
  Keep the ordinary `g(s,z)` MLP, 4096 fixed normal OT integration sites,
  256 fresh teacher latents/Gaussian centers with one action each (stratified),
  importance resampling 256 to 16 BEFORE OT, and 4096x16 OT assignments from
  the persistent state-conditioned dual MLP. Evaluate actor Gaussian outputs
  only at the selected 16 training latents; do not reinstate full-H mixture
  actor density. Teacher correction is `softmax(Q/T-log q)` once.
  Source sampling uses detached `(1/H)/sum_j(P_ij)` without clipping or
  self-normalization. Freeze map and critic parameters, but retain BOTH Q
  and assignment gradients through the newly generated actor action.
  Preserve dual ReLU 256x2->H, zero output init, Adam 1e-4, one dual Adam step
  per actor step, and the same pre-update potential for actor and analytical
  `sum_j(P_ij)-1/H` dual gradient. GMM shares f across identical-state lanes;
  RL needs f(s). Preserve dual parameters and Adam state between updates.
  Fresh Sinkhorn100 is an explicit solver control; its source-importance<=K
  bound does not apply to the persistent default. Monitor variance and finite
  gradients. Teacher sigma floor .05 never enters actor or TD density.
  Critic uses soft TD with alpha=T and a fresh 16-component self-inclusive
  marginal density estimate (IDAC path), not the 4096 OT bank density.
  Collection/evaluation retain the continuous normal latent prior. Preserve
  v5 LR, UTD, frequency, network, sigma limits and paired epsilon=0 evaluations.
  Existing conditional GMM and epsilon runs remain records of this objective.
  Archive the separate marginal-SAC comparison; never silently resume a
  different objective as the same experiment. See CONDITIONAL_RESTORE_KO.md.
  Preserve frozen sources, historical artifacts and external baseline original
  implementations. GMM validation precedes full RL campaigns. Restoration
  alone authorizes neither new training, a full RL campaign nor a push.
  The following sections describe historical profiles.

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
