# v3 teacher-floor1 acquisition and retention confirmation

The authorized active goal permits bounded reward/hyperparameter experiments
without algorithm changes. Review of the completed d259301 teacher-floor screen
found44/100 direct right successes and0left, with11left entries; mu-only80/100
right successes. Full training replay contained no goal-radius/terminal visits.
At258304total, each of256environments had only1009transitions including warmup.
This motivates testing acquisition and retention over more collected episodes;
it does not establish a mechanism or claim that both routes will succeed.

Register ONE fresh seed0 v3 run, teacher_std_floor1, T3, discount.999, the same
fixed-origin-normalized geodesic progress100 reward, bonus0, stepcost0 and
NovelD OFF. Change only budget/id/hypothesis from the completed short manifest.
This is not a resume, independent seed, or automatic extension of an old queue.
All model, actor sigma[-5,-1]/initial-1, random latent,N=M64,beta1,mean-init1,
256x3GELU,Adam actor3e-4/critic5e-4,tau.005,batch4096,replay1M,256env,
8updates/256,warmup8192,DACERH/d+.7/500,alpha.27/alphaLR.03/noise.1/GMM3/200
remain unchanged. All seven pinned computational files must match2564b59.
No environment, physics, reward implementation, runner, loss, TD, optimizer,
sampler or regulator edits. v4 teacher-floor screens are preserved, not extended.

Max1M post-warmup under original strict-greater counting:1008384total transitions,
31256learner updates,63DACER updates.40direct and40mu-only episodes each50k;
100final each mode and full checkpoint. Original identical v3 full starting
state and native700episode limit. Direct uses randomlatent+conditional sigma;
external DACER noise and intrinsic reward remain absent from evaluation.
Report failed episodes in every denominator. More corridor entries alone do
not satisfy the successful multimodal goal.

Inspect every50k; from500k, stop if left entries are at most1/40 and left
successes remain0 for three consecutive direct-policy evaluations. Preserve
the stopped-run evidence. Runtime/numerical failure holds the job without retry.
No automatic extension beyond the cap. Compare the shared pre250k policy/data
prefix where applicable, and explicitly report any mismatch before treating
the short and long runs as the same trajectory.

Commit/push/share/freeze before real8448transition/8batch4096update preflight
and launch. Require actual manifest equality exceptid/hypothesis/steps against
the completed d259301 run, initial model/critic hashes and runtime config checks.
Use one independently unlocked180GPU; share source with199, launch no199job.
No4090, cancelled job resume, new seed, new automation or algorithm changes.
W&B OptiQ/antmaze, group equal to campaign basename. Preserve every old source,
checkpoint and evaluation; record local report source separately.
