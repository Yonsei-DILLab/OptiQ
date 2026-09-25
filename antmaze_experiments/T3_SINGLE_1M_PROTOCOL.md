# iBOLT AntMaze T3, 2026-09-26

User requested our method, v1-v4, T3,1M. Four fresh seed0 runs.
Keep unchanged basic Direct-GMM/TRG learner:N=M64,beta1,256x2 actor/critic,
log_std[-5,-1]/initial-1,LR3e-4,gamma.99,ordinary TD,no DACER/no intrinsic.
Dense negative nearest-goal Euclidean reward, no added step penalty/goal bonus.
Single environment,batch256,UTD1 after10K random warmup,1M total/990Kupdates.
Native v1 random training start,v2-v4 native fixed training start.
Fixed full original origin evaluation every25K,100 episodes/mode:
native=random z/mu only; policy=random z plus conditional noise.
Final evaluation also zero-z mode, clearly separately labeled.
Save intermediate policy/critic evaluation checkpoints, trajectories NPZ;
final full state/replay and integrity checks. Preserve original500/700 horizons.
vast4 GPUs0-3 map to v1-v4. Prior Point runs completed; do not alter them.
W&B OptiQ/jaehun-antmaze, group ibolt-antmaze-t3-single-1m-20260926.
Each job runs existing8actual-update preflight, audit, serialization, evaluation
before main training; failure stops that job. No legacy queues restarted.
