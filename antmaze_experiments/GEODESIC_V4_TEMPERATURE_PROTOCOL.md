# v4 teacher-temperature-only screen

The active goal permits bounded reward/hyperparameter screens without changing
the algorithm. The original-geodesic gamma.999/T1 candidate lost its upper
route: direct40/40 lower at350208 and400128transitions, lower39/none1 at450048,
with no successes. Both origin-to-goal geodesic distances are17.6568562495m,
so the unequal initial goal distances found in v3 do not explain this case.

Test fresh v4 seed0 at teacher T3 and T10. Hypothesis: softer Q weighting may
retain minority actions as estimated values diverge. Higher T can also prevent
goal acquisition; neither entropy nor two corridor entries counts as success.
The completed v4 original-geodesic250k/T1 condition from438907f is the paired
same-budget control. The newer1M-budget T1 run has exactly matching shared
checkpoint samples, but is not another training seed.

Only teacher temperature changes. Keep the ORIGINAL geodesic center-distance
progress reward100*(d_current-d_next), bonus0, step cost0 and NovelD OFF.
Do not use the v3 normalized reward here. Keep gamma.999, DACER H/d+.7/500,
initial alpha .27,alpha LR .03,noise scale .1,GMM3/200samples; keep actor/twin
critics256x3GELU,LR3e-4/5e-4,Adam,tau.005,beta1,randomz,N=M64,mean scale1,
log sigma[-5,-1]/initial-1.256env,batch4096,8updates/256,warmup8192,replay1M.
Budget250k post-warmup is258304total transitions and7816learner updates.

Seven core algorithm files remain byte-identical to2564b59. No reward, runner,
optimizer, TD, NLL, sampler, upstream physics, reset or termination changes.
Commit/push/freeze before real per-job8448/8update preflight. Require unchanged
initial actor/critic hashes, source/config/replay reward/checkpoint verification.
Failing jobs hold pending jobs; no automatic retry or budget extension.

Register only on180. Preserve current v3 normalized-reward learners and the
v1/v4 geodesic1M jobs until a separately documented active-goal screen stops a
failed candidate. Use existing GPU locks and independent backfill. Do not
stop a shared controller while its v1 job is still running.199 source-only sync
remains pending its SSH outage. No4090 AntMaze use or cancelled-job restart.

Evaluate40episodes each50k and100final per direct/mu-only mode, at the original
v4 fixed full state. No external DACER noise. Store policies and final full
state. Compare exact-step raw trajectories, successful goal/route counts,
failures and later retention before any positive claim. One training seed.
