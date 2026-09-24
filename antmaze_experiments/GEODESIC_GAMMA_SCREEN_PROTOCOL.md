# Wall-aware progress at gamma .999

Active goal: preserve the algorithm and test existing reward/hyperparameter
settings to obtain sustained multiple successful AntMaze routes.

The completed v3 Euclidean gamma .999/T1 screen retains54left/33right entries
in100 direct-policy episodes, but no goals. The fresh longer same-seed run is
preserved. Three otherwise idle180 slots test the existing geodesic reward
profile on v3/v4/v1, each fresh seed0 and250k post-warmup transitions.

## Controlled change

Compare with each maze's `gamma999` condition in source
eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45. Only the distance metric changes:
`100*(d(current)-d(next))`, where d is nearest-goal XY visibility-graph
geodesic distance. No success bonus, step penalty or NovelD. Goal success and
termination remain unchanged, terminal distance is not overwritten with zero.
The existing geometry uses point-agent XY paths, margin1e-6m and no body
inflation. It is not a Habitat navmesh or full-body collision-free path oracle.
v3's two goals are not equidistant at the start, so this does not claim to
equalize goal values or prescribe balanced route counts.

Keep gamma .999, T1, DACER H/d+.7 (total5.6), interval500, initial alpha .27,
alpha LR .03, noise scale .1, GMM3/200 samples. Keep actor/twin critics256x3
GELU, actor/critic LR3e-4/5e-4, Adam, tau .005, random z, N=M64,
mean-head scale1, log sigma[-5,-1]/initial-1, beta1. Use256env, batch4096,
8updates/256transitions, warmup8192 and replay1M. Native accounting gives
258304total transitions,250112post-warmup and7816learner updates.

No algorithm implementation changes: verify the same seven core files listed
by register_horizon_temperature.CORE_FILES against2564b59. No vendor, reset,
physics, reward implementation, optimizer, TD, NLL or sampling code changes.
The launch manifest differs from the matched screen only in job identity,
hypothesis and the existing reward profile/specification.

## Execution and reporting

Source must be committed and pushed before any preflight. Register only on
vast-heechan-180, under antmaze-optiq-geodesic-gamma999-250k-s0-20260925.
Respect the live1M campaign's reservations and existing GPU locks; fill other
slots independently. No cancelled job is resumed and no4090 work is allowed.
199 is source-only; if its SSH transport remains unavailable, record delayed
source synchronization explicitly, and do not infer learner failure.

Each job runs its own8448transition/8real-update preflight with full checkpoint
readback and replay reward verification. A failure holds pending jobs and
preserves live jobs; no automatic retry. W&B OptiQ/antmaze.

Evaluate each50k with40 episodes per mode, retain policy checkpoints, and save
100final episodes plus full state. v1 uses original random starts; v3/v4 use
original fixed full state. Direct policy includes random z and conditional
sigma; random-z mu-only is separate. No external DACER noise in evaluation.

Report raw successful routes and failed episodes, closest distance, retention
at later checkpoints and exact-budget comparisons. Route entry alone is not
goal completion. All runs use seed0; paired settings and the longer same-seed
run are not independent training replications. Do not promote an isolated
successful checkpoint to a sustained-multimodality claim.
