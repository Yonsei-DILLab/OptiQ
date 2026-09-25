# PointMaze diagnostic

Explicit user approval: replace Ant with Point while retaining our algorithm.
New environment, not a claim of reproducing Ant physics or benchmark performance.
v1 walls: interior box[-6,-2]x[-2,2], outer free bounds(-10,2)x(-6,6).
Goal(-8,0), radius.5; reward minus remaining Euclidean distance; no bonus,
step penalty, intrinsic reward, DACER noise or soft backup.
Observe xy; action in[-1,1]^2 produces .2*action movement. Ten collision
substeps reject movement into walls (no axis-order bias or tunnelling).
Training starts uniform[-2,2]^2; evaluation always starts(0,0), horizon500.
Actor collection retains random z and conditional noise. Evaluation uses
random z, conditional noise OFF (mu only), redrawn every step.
Existing Direct-GMM/TRG learner unchanged:256x2, batch256, UTD1 after10K warmup,
N=M64, beta1, log_std[-5,-1]/initial-1. Seeds0, T.5/1,250K total each.
Every5K:100 fixed-start rollouts, trajectories PNG/NPZ,4096 s0 mu actions,
actor/critic checkpoints (not full replay resumable state). Evaluation preserves
training random streams. Test symmetric upper/lower scripted paths and collision;
perform4 real updates, serialization, rollout before main launch.
W&B OptiQ/jaehun-pointmaze; idle vast3 GPUs2/3. Existing runs untouched.
