# OptiQ AntMaze critic-dynamics diagnostic

User approval 2026-09-25: run a small, fresh diagnostic of the observed gap
between predicted Q and realized return, and of subsequent route concentration.
This is a bounded hypothesis test, not a final performance benchmark. Run only
OptiQ, v3/v4, seed0, with the six conditions below: twelve policies total.
Keep existing training, including an existing v4 learner if still active,
unchanged. Never resume old cancelled queues. Do not use vast1 or any RTX4090:
that host is reserved for GMM40.

Campaign and W&B group:
`antmaze-optiq-critic-dynamics-500k-s0-20260925`.
W&B project is `OptiQ/antmaze`. Reference source is
`f953d28456d3800860dddb9b9cb91b6bd520ae00`; historical reference campaign is
`antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2`. Every diagnostic policy,
including both controls, starts fresh from the newly committed frozen source.

## Conditions and unchanged settings

| dynamics_profile | Reward multiplier | Teacher T | Critic target tau | Critic LR | Actor update delay |
|---|---:|---:|---:|---:|---:|
| control | 1 | 1 | .005 | 5e-4 | 1 |
| scale02 | .2 | .2 | .005 | 5e-4 | 1 |
| ema01 | 1 | 1 | .01 | 5e-4 | 1 |
| criticlr2 | 1 | 1 | .005 | 1e-3 | 1 |
| actordelay2 | 1 | 1 | .005 | 5e-4 | 2 |
| scale02ema01 | .2 | .2 | .01 | 5e-4 | 1 |

The runtime condition definitions in `dynamics_profiles.py` are the single
source for the manifest and command-line profile. The controller passes
`--dynamics-profile <name>`; this is not an unrecorded environment-variable
override. Conditions other than `scale02` and `scale02ema01` use
`progress100_euclidean_no_step_no_bonus`: r=100*(d_current-d_next).
The two scaled conditions use the explicit
`progress20_euclidean_no_step_no_bonus` profile: r=20*(d_current-d_next).
Here d is nearest-goal Euclidean XY distance. Success bonus=0, step cost=0,
NovelD OFF. There is no gamma inside the reward. The actual terminal distance
is retained, success within 0.5m terminates, and time limits bootstrap.
Physics, maps, observations, action bounds and resets are unchanged.

All conditions retain Direct GMM/TRG, random latent, N=M64, log sigma[-5,-1]
with initial-1, mean-init1, beta1, actor and twin scalar critics256x3 GELU,
actorLR3e-4, Adam, gamma.99, replay1M, warmup8192,256 parallel environments,
batch4096 and eight critic updates per256 collected transitions. DACER is OFF;
neither training nor evaluation adds DACER behavior noise. Critic backup is
plain TD, with the current semi-implicit actor and target critics; no entropy
backup is added. Actor delay2 intentionally halves actor updates while keeping
collection and critic-update counts fixed. No sigma, replay-size, clipping,
intrinsic-reward or reward-shaping additions are part of this diagnostic.

The reward-and-temperature scaling condition preserves the ideal Q/T ratio
if learned Q scales exactly with reward. It is therefore a test of learning
and numerical dynamics, not an intentional increase of the optimal policy's
entropy. Finite-time function approximation and optimizer behavior need not
obey that ideal scaling identity. The combined condition checks whether the
two changes interact; do not attribute its outcome to either change alone.

## Budget, evaluation, and diagnostic records

Nominal main budget is500,000 post-warmup environment transitions. Preserve the
upstream strict-greater stopping rule and256-transition collection blocks:

```
warmup                  = 8,192
post-warmup transitions = 256 * (500000 // 256 + 1) = 500,224
total transitions       = 508,416
critic updates          = (508416 - 8192) / 256 * 8 = 15,632
actor updates           = 15,632; actordelay2 = 7,816
```

Both mazes use this diagnostic budget; do not substitute their usual4M/5M
budgets or extend a run automatically. Main evaluations and evaluation-only
policy checkpoints are scheduled every100,000 **total collected transitions,
including warmup**, rounded up to a256-transition block. The five intermediate
checkpoints are100096,200192,300032,400128,500224. The final508416 result is
separate. Record total and post-warmup counters so their meanings are explicit.

Each intermediate checkpoint uses40 episodes per native/direct mode; final
evaluation uses100 per mode. Evaluation resets match training: v3/v4 start at
the original fixed origin, posture and velocity, not the previous random XY
override. Native=mu-only with random latent; direct policy=random latent plus
conditional sigma. Use direct policy for route diversity, and preserve native
as a separate output. Neither mode includes NovelD reward or external noise.
Retain final full replay, model, target, optimizer, RNG and simulator state.

Keep diagnostic raw state/action arrays, online and target Q predictions,
teacher candidates and weights, Monte Carlo returns and route labels with
checkpoint and reporting-source provenance. Probe inference must not update
training parameters, optimizer, replay, counters or training RNG; verify
immutable learner-state identities, keys and update counters before/after.
Intermediate checkpoint serialization has separate SHA256/readback checks.
Compare online and target critics on identical state/action pairs. Across
conditions, t0 shares the original full start state but policy actions differ;
later landmarks and replay samples generally contain different visited states.
Retained arrays allow later common-state/action checkpoint probes, but such
cross-checkpoint probes are not automatically executed by this campaign.

Primary questions are whether online/target value lag or MC mismatch improves,
whether actor route usage or teacher concentration changes, and whether both
routes remain represented as training proceeds. Success and return are
secondary diagnostic context. Report failures and single-seed uncertainty.
Current-policy MC is a finite-sample estimate, not exact ground truth. A later
critic evaluated on an old state/action bank predicts continuation under a
later policy: its difference from the old policy's RTG is not automatically
critic error. Teacher mass induced by a forced first action and subsequent
actor continuation is not a rollout of a teacher policy at every step.

Replay diagnostics use an independent fixed-key sample every25k total steps,
with eight fresh next-policy actions per state, and report TD fitting residual,
target tracking contribution and twin-min contribution. Evaluation traces use
the actual subsequent sampled actions and a64-action timeout boundary estimate;
raw finite-horizon returns remain separate from bootstrapped returns. The latter
uses the online critic and is not independent ground truth. Compare errors after
division by reward_multiplier, so a fivefold change of units cannot appear as
better fitting. Diagnostic scalar summaries are also logged to W&B.

## Scheduling, preflight, and provenance

| Eligible order | vast-heechan-180 | vast-heechan-199 |
|---:|---|---|
| 1 | v3 control | v4 control |
| 2 | v4 scale02 | v3 scale02 |
| 3 | v3 ema01 | v4 ema01 |
| 4 | v4 criticlr2 | v3 criticlr2 |
| 5 | v3 actordelay2 | v4 actordelay2 |
| 6 | v4 scale02ema01 | v3 scale02ema01 |

Each host has three v3 and three v4 jobs, and one job for every condition.
Conditions are spread across both hosts; this reduces host imbalance but does
not establish a multi-seed, fully counterbalanced hardware experiment. Start
controls first on both hosts. Use up to the existing eight5090 GPU slots,
respect all GPU locks, and immediately backfill any free slot every2seconds.
There is no all-preflight, all-maze or all-condition completion barrier.

Every job runs its own fresh8,448-transition preflight with real256-env
collection and batch4096 updates, then starts fresh main training only after
its proof passes. Preflight has eight critic updates; actordelay2 must have
four actor updates and all others eight. Verify profile settings, actual
reward values in replay, actor/critic optimizer learning rates, tau, update
counters, initial-parameter agreement under seed0, checkpoint readback and
evaluation sampling. Preflight saves one evaluation-only policy checkpoint.
Do not reuse a different condition's proof or silently relaunch failures.

Commit/push/share exact code, conditions, registration and this protocol
before preflight or main training. Registration requires a clean immutable
`/home/heechan/OptiQ-ops/sources/<full-commit>` checkout and records the full
commit in manifests and submitted jobs. Reuse the existing controller and
W&B sync sidecar; autostart=false, autorestart=false. Refuse an existing
campaign directory or service name. A failure holds pending jobs on that
shard while preserving other live jobs. Preserve existing frozen experiments,
cancelled queues and unrelated workers. No4090 AntMaze work is authorized.
