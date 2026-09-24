# OptiQ Euclidean progress, no step penalty or success bonus

User approval 2026-09-25: stop all running/pending geodesic AntMaze jobs on
180, 199 and vast1, preserving frozen sources, checkpoints, logs and status
snapshots. Launch exactly four fresh OptiQ policies, v1-v4 seed0.

Reward: r = 100 * (d(current) - d(next)), where d is nearest-goal Euclidean
XY distance. Step penalty=0, success bonus B=0, NovelD OFF. This is a new
progress100_euclidean_no_step_no_bonus profile; historical profiles are unchanged.
There is no gamma multiplier inside the reward. Terminal distance is the actual
distance to the goal center, not zero. Upstream success still terminates within
0.5m; timeout bootstrapping and physics remain unchanged. A stationary transition
receives zero, approaching a goal yields positive progress and retreat negative.
Do not describe this as the earlier dense negative-distance reward.

Only the distance metric changes relative to the geodesic no-step/B0 learning
control (source4d757159ef84e849250a6d5111f9931b2b392b21). T1, beta1, DACER OFF,
actor and twin critics256x3 GELU, mean-init1, random latent, N=M64,
log sigma[-5,-1]/initial-1, actorLR3e-4/criticLR5e-4, Adam, gamma.99,tau.005,
replay1M,256env,batch4096,8updates/256transitions,warmup8192 are preserved.
Native budgets v1/v2 3M,v3 4M,v4 5M exclude warmup and retain strict-greater
stopping. No baselines, additional seeds or automatic restarts are authorized.

Evaluation matches training resets: v1 upstream random XY[-2,2], v2-v4 original
fixed XY[0,0], posture and velocity. Set eval_starts=upstream. Evaluate every250k
with40 episodes each native/direct mode and an evaluation-only policy checkpoint;
final100 episodes per mode/reset and full replay/model/optimizer/RNG state.
Direct policy includes conditional sigma; native is random-z mu-only. Neither
mode adds external DACER noise or intrinsic reward. Keep old random-start data
separate when comparing the geodesic control.

Commit/push/share source before preflight or training. Host180 runs v3/v4;
host199 runs v1/v2. Share source to vast1 without launching extra runs. Independent
per-slot scheduling every2seconds; every job must pass the real256env/batch4096
8448-transition/8-update preflight, replay reward check, checkpoint readback and
settings validation before its fresh main run. Failure holds pending jobs.
Use existing supervisor and GPU locks. W&B OptiQ/antmaze, group
antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2.

The initial campaign without -r2 (sourcecab48faf07d41a86886355bfd6bef538fff29432)
failed in configuration creation: reward_description did not recognize the new
no-cost Euclidean profile. No warmup collection, learner updates or main runs
started. Preserve those preflight logs and failure manifests; use a new committed
source and separate -r2 campaign after fixing and testing the logging dispatch.
