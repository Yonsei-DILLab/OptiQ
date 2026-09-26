# PointMaze Simple entropy sensitivity, 2026-09-26

The user requested extra parameter experiments after MFPO and MEOW each reached
only one Simple goal in all 500 final rollouts despite 100% success. These six
fresh seed0 jobs test an entropy-setting explanation, not a changed algorithm.

MFPO changes only target_entropy_coeff from the preserved -0.5 control to
0, +0.25, +0.5 per action dimension (total targets 0, +0.5, +1 for 2D).
Native automatic temperature tuning, initial alpha .01 and temp LR3e-4 remain.
The existing Simple checkpoint alpha is .0163673 at1M, not a fixed .01.
Native flow log-density is learned before a final hard action clip: it is not
an exact entropy of clipped actions. Log boundary saturation as well as entropy.

MEOW changes only fixed alpha from the preserved .2 control to1,3,10.
No entropy autotuning is introduced. Native flow, base-prior log-sigma[-5,-.3],
Adam1e-3, tau.005, gradient clip30 and Q/V parameterization remain. The prior's
sigma is not the final transformed action sigma. Alpha scales both Q/V and
entropy, so this sensitivity includes optimization scaling inherent to MEOW.

All jobs keep the original DrAC Simple map and sparse+100 reward,256env,
batch4096,16updates/256transitions,replay1M,warmup8192, gamma.99 and native
networks. Stop at1,000,192 transitions (62,000 updates), with200k/200episode
intermediate and500episode final direct-policy evaluations and checkpoints.
MFPO uses sample_actions, not best-of10 selection; MEOW samples its flow.
OptiQ remains fresh-random-z mu-only, but no OptiQ training is added here.

Record native optimizer diagnostics every1000 updates, and512 native-density
samples from the same reset-state distribution at each evaluation. Probes use
separate saved/restored RNG, perform no updates, and preserve the training RNG.
Save alpha, native entropy estimate, Q scale, action covariance and clipping.
Primary comparison: goal counts including failures, goals reached, concentration,
and trajectories. Success alone and local action entropy do not prove route
multimodality. This is one seed; do not call tuning evidence a robust benchmark.

Commit/push exact code/plan before starting. Export pinned MFPO/MEOW dependencies
and hash the frozen source. Use only idle46 GPUs0,2,4,5,6,7 through shared GPU
locks; leave running Hard T5/T10 and199 DIPO untouched. Per-job8448-transition /
16-update preflight must pass before main. Independent workers; no automatic
retry. Preserve existing controls and all running/frozen sources.
