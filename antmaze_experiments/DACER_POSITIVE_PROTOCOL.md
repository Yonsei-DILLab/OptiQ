# Positive DACER entropy targets on AntMaze

User approval, 2026-09-25: run OptiQ v1-v4 with per-action-dimension entropy
targets **+0.1, +0.5, +0.7, +0.9** and regulate every **500 learner updates**.
Use the discussed bounded 500k post-warmup diagnostic budget, seed0, T=1.
These are positive targets: for eight actions their totals are 0.8, 4, 5.6, 7.2.
500 denotes the regulator interval, not a fifth entropy target.

## Fixed learning settings

- Reward: `100*(d(current)-d(next))`, nearest-goal Euclidean distance; no
  success bonus or step penalty. NovelD OFF. Preserve success termination,
  time-limit bootstrapping, geometry, physics and reset distributions.
- Direct GMM/TRG, actor and twin critics 256x3 GELU; mean-head scale1;
  random latent N=M64; log sigma[-5,-1], initial-1; beta1.
- Actor Adam3e-4, critic Adam5e-4, tau.005, existing policy delay1,
  replay1M, batch4096, 256 environments, 8 learner updates/256 transitions.
- DACER ON, behavior-only. Initial alpha.27, alpha LR.03, noise_scale.1,
  3-component full-covariance GMM, 200 samples/state, entropy seed42.
  The entropy-probe replay batch remains the existing learner batch4096.
  No actor entropy loss, Bellman backup, clipping threshold or estimator change.
- 8192 warmup transitions. Upstream `global_steps > max_step` accounting gives
  500224 post-warmup / 508416 total transitions and 15632 learner updates.
  Regulator fires at learner counts 0,500,...,15500: 32 updates, next16000.
  This cadence is approximately every16k interactions after warmup.

## Evaluation and interpretation

Every100k total transitions (rounded to a256 block), save an evaluation-only
policy and evaluate40 episodes/mode. Final evaluation is100 episodes/mode;
save full replay/model/optimizer/target/DACER/RNG/simulator state at final.
v1 uses its native random reset; v2-v4 use their original fixed full-state
start, matching training. Direct-policy trajectories include conditional sigma;
mu-only/native and zero-z are separate. Evaluation adds no external DACER noise.
Intermediate checkpoints retain all fields needed for additional trajectory work.

Preserve every entropy update in `dacer_regulator_history.jsonl`, with estimator
value, target, noise std before/after the update and probe clipping fraction.
W&B also receives behavior-action clipping fraction and noise std. These are
observational logs; they consume no extra randomness or optimizer updates.
GMM joint entropy is a proxy, not exact clipped-action or trajectory entropy.
A larger target need not increase policy route diversity; evaluate the saved
direct-policy routes, failures and state coverage separately from training noise.

## Registration and provenance

Campaign: `antmaze-optiq-dacer-hpos-i500-500k-s0-20260925`.
`python -m antmaze_experiments.register_dacer_positive --shard 0 --host vast-heechan-180`
and shard1/host199. Eight jobs per host, each includes all mazes and all targets.
Both hosts' first four jobs cover v1-v4; subsequent jobs backfill independently
every2seconds. Only the eight5090 GPUs are used. Do not use vast1/4090.

Commit/push/share exact source before launch. Register only from
`/home/heechan/OptiQ-ops/sources/<full SHA>` with a clean tracked tree. Supervisor
services are not auto-restarted. Each job first validates its real batch4096
updates, target sign/total/interval, reward replay and checkpoint readback.
Failure holds pending jobs and preserves live jobs; no automatic restarts.
Preflight8 updates fire the regulator once and leave next_update500.
Controller audits the complete regulator history and counts at final.

Use W&B `OptiQ/antmaze`, group=campaign. Preserve all previous immutable sources,
completed and cancelled results; this launches no baseline or extra seed.
Defaults for other campaigns remain the original10000 regulator interval.
Reference DACER-off source: `f953d28456d3800860dddb9b9cb91b6bd520ae00`;
immediate parent: `23603a7e7696aa64e8e49a38b986d6b430b6d6f3`.
