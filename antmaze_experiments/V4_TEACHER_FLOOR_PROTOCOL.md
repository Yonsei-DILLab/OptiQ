# v4 teacher-only proposal width screen

The active goal permits bounded reward/hyperparameter tests while preserving
algorithm structure. Original-geodesic gamma.999/T1 reaches only the lower
successful route by600k; the increasing T1->3 candidate loses the upper route
and has zero direct successes at700k through900k. That candidate was screened
and stopped, preserving completed v1 and all old snapshots. The separate v3
teacher-floor screen shows some majority-route acquisition improvement but no
minority success through250k. This does not predict a positive v4 result.

Run two fresh v4 seed0 screens at existing teacher_std_floor .5 and1.0. Compare
with the archived original-geodesic250k control, source
438907f3a5ef681d6cde9012a66c7336fa642546, whose floor is exp(-5). Retain T1,
gamma.999,100*nearest-geodesic-distance decrease, bonus0,stepcost0,NovelD OFF,
DACER H/d+.7/interval500,alpha.27/alphaLR.03/noise-scale.1/GMM3/200.
Keep256x3 actor/critics,Adam actor3e-4/critic5e-4,tau.005,random normal latent,
N=M64,beta1,mean-init1,log sigma[-5,-1]/initial-1,256env,batch4096,
8updates/256,warmup8192,replay1M, native physics/reset/horizon/termination.
No reward, loss, target, optimizer, sampler, regulator or pinned core changes.

Only the existing teacher floor changes. Existing box-truncated candidate
sampling and density use the same effective scale; actor sampling and evaluation
sigma do not acquire this floor. Do not describe it as removal of actor sigma
bounds. Finite importance sampling variance or critic extrapolation can worsen;
success on just one side is not the goal.

Max250k post-warmup =258304total/7816updates.40episodes/mode each50k and100final,
intermediate evaluation policies and final full state. v4 original identical
full start, direct=random latent+conditional sigma; mu-only separate; no external
DACER noise or intrinsic reward in evaluation. Inspect both successful routes,
failed episodes and final-distance distribution. No automatic extension/retry.

Use only independently unlocked vast-heechan-199 GPU slots after checking its
guide and active workload. Its transport recovered; do not resume any old
unknown/cancelled job or use4090. Commit/push/share/freeze source before real
8448transition/8batch4096update preflights and main launch. Validate sampling/
density, resolved actor floor and unchanged initial parameters against the actual
archived v4 control, then actual source/config/GPU-PIDs and checkpoint readback.
W&B OptiQ/antmaze, group equal to campaign basename. Preserve all artifacts.
