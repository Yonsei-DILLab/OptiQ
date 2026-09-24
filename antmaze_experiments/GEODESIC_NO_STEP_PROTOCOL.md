# OptiQ geodesic progress only, no step cost or goal bonus

User approval 2026-09-24: cancel all running and pending Euclidean entries in
the progress100 factorial. Preserve existing geodesic learners and queued jobs,
their original source0751e86 and saved results. Launch exactly four fresh OptiQ
policies, seed0, in priority order v3/v4/v1/v2 using the released 5090 slots.

Reward is r=100*(d(current)-d(next)), nearest-goal XY visibility-graph geodesic.
B=0 and step penalty=0. This removes only the -1 cost relative to the existing
geodesic-no-bonus control. No gamma inside reward. Goal-center distance remains
actual at terminal transitions; success still terminates within0.5m. Timeout
bootstrap, wall margin1e-6m and original MuJoCo physics are unchanged.
The separate progress100_geodesic_no_step_no_bonus profile does not change any
historical reward profile. Stationary transitions now receive zero, retreat
negative progress. This is an ablation, not a guarantee of path diversity.

T1,DACER OFF,NovelD OFF,beta1,actor/twin critics256x3GELU,mean-init1,randomz,
N=M64,log sigma[-5,-1]/initial-1,actorLR3e-4/criticLR5e-4,Adam,gamma.99,tau.005,
replay1M,256env,batch4096,8updates/256transitions,warmup8192. Native budgets
v1/v2 3M,v3 4M,v4 5M exclude warmup and preserve strict-greater stopping.
Evaluate every250k:40 random-start episodes per native/direct mode and policy
checkpoint; final100 per mode/reset and full replay/state. Primary trajectories
are random-start direct policy, native mu-only supplementary. No extra noise.

Host180 runs v3/v4;host199 runs v1/v2. Independent2second backfill, per-job real
256env/batch4096 preflight8448transitions/8updates with replay reward validation
and checkpoint readback before fresh training. Failure holds pending, no retry.
Commit/push/share exact source before any new preflight or main launch.
W&B OptiQ/antmaze, group antmaze-optiq-geodesic-no-step-B0-T1-s0-20260924.

Selective cancellation uses preserve_geodesic.py: freeze scheduling, save old
state, kill only the original controller PID, terminate only Euclidean process
trees, cancel pending Euclidean entries, and adopt geodesic wrappers using PID
creation times. Keep the original manifest/training source untouched; separate
sidecars record cancellation and new controller commit. Pending v2 geodesic
controls on vast1 start with their original source. No learning restart.
