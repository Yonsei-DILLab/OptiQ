# v3 gamma .999 / T3 longer candidate

Active-goal follow-up, with no algorithm changes. The completed250k screen
(source eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45) has direct-policy entries
left20/right80, successfulright33/100. Mu-only hasleft17/right79/none4 and
successfulright49/100. Its remaining left entries do not establish successful
multimodality. The separate longerT1 control has already lost right entries
by350-450k, so test the existingT3 setting's later acquisition and retention.

Register one fresh v3 seed0 1M post-warmup run on an unlocked180 GPU. It starts
from scratch with identical gamma999_temp3 settings; it is neither a resume
nor an independent seed. No previous source, result, live job, cancelled queue
or199 job is changed. Keep the original short prefix as a reproducibility check.

Only budget and job identity differ from the short T3 screen. gamma .999,T3,
DACER H/d+.7,total5.6,interval500;100*Euclidean distance decrease,B0,stepcost0,
NovelD OFF. Preserve256x3 actor/critics,3e-4/5e-4LR,Adam,tau.005,random z,
N=M64,mean-init1,log sigma[-5,-1]/initial-1,beta1,256env,batch4096,8updates/256,
warmup8192,replay1M. Native accounting gives1008384total,1000192post-warmup,
31256learner updates and63regulator updates.50k evaluation/checkpoints,40
episodes/mode each and100final; original fixed full state, direct policy and
random-z mu-only separate, no external DACER noise in evaluation.

Commit/push and freeze before actual preflight/main runs. Verify the seven
core files against2564b59 and each job's real8448step/8update preflight,
replay rewards and checkpoint readback. Use supervisor, respect GPU locks,
preserve currently running work and do not use4090. Source-only199 sharing
is pending SSH recovery and must be recorded honestly. W&B OptiQ/antmaze.

Inspect minority successful routes and final goal distances at500k/750k/1M.
Do not claim success from entry counts, pool checkpoints as one policy or
generalize the historical successful v1 same-state probe to v3/v4.
