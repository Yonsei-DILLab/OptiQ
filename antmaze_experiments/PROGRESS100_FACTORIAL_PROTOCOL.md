# OptiQ AntMaze 100x progress, unscaled terminal bonus

User approval 2026-09-24: stop the previous 16-policy progress factorial and
restart the same Euclidean/geodesic x success bonus ON/OFF x v1-v4 grid,
seed0. Reward is exactly r=100*(d(current)-d(next))-1+B. B remains 10 for
all goals except v2(-8,8)=20 when ON; B=0 when OFF. B is NOT multiplied100.
No gamma inside reward. Terminal goal-center distance is retained; radius0.5,
success termination and timeout bootstrap are unchanged, even when B is OFF.

Profiles progress100_euclidean, progress100_euclidean_no_bonus,
progress100_geodesic, progress100_geodesic_no_bonus are distinct from historical
progress_* profiles, whose numerical behavior remains unchanged. Geodesic remains
the original XY visibility graph with1e-6m wall margin, not body-space planning.
See PROGRESS_REWARD_PROTOCOL.md for geometry and PROGRESS_FACTORIAL_PROTOCOL.md
for all unchanged learning/evaluation settings.

Constant T1, DACER OFF, NovelD OFF, beta1,256x3 actor/twin critics,GELU,mean-init1,
random latent,N=M64,log sigma[-5,-1]/initial-1,actorLR3e-4,criticLR5e-4,Adam,
gamma.99,tau.005,plain TD,no clipping or normalization. Replay1M,256envs,
batch4096,8updates/256transitions,8192warmup. Native post-warmup budgets remain
v1/v2 3M,v3 4M,v4 5M, with the original strict-greater stopping convention.
Every250k evaluate40 random-start episodes/mode and save policy checkpoints;
final100/mode/reset and full state/replay. Keep fixed-start supplementary.

Three four-GPU hosts: vast-heechan-180/199 (5090) and vast1 (4090). Each5090
host has22M native transitions,4090 host16M. Independent backfill every2seconds,
no cross-maze/preflight barrier. Each job performs its own real256env/batch4096
8448transition/8update preflight, then fresh main training. No auto retry.
Additional server requires adequate disk and compatible runtime before launch;
pre-existing NM/ablation assets and active processes must remain preserved.

Commit, push and share identical frozen source before any learner preflight.
Project OptiQ/antmaze, group antmaze-optiq-progress100-2x2-T1-s0-20260924.
Record GPU/runtime and source commit per host. Previous campaign cancellation
is recorded separately; preserve its frozen sources, logs and checkpoints.

Replay audit reconstructs rewards from float32 saved XY while physics used
float64. Absolute tolerance is2e-5*progress_scale (0.002 for new profiles),
with rtol2e-6 and existing terminal-radius boundary logic. Record tolerance
in the proof. This is storage quantization handling, not reward clipping.
Validate exact forward/rest/retreat and bonus magnitudes, unchanged physics,
autoreset endpoints, timeout bootstrap, and checkpoint readback before main jobs.

100x is a scale-matching hypothesis, not a guaranteed exploration improvement.
B stays unscaled, so this is not100 times the entire previous reward. Report
trajectory/coverage/success rather than directly comparing raw reward totals.
Each setting has one seed; GPU architectures differ and timing is not a fair
algorithm comparison. Preserve all cancelled and historical campaigns.
