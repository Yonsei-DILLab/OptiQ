# Existing teacher proposal floor: bounded v3 screen

This experiment is part of the active reward/hyperparameter goal. It does not
change the learning algorithm, physics, observation, actor sampler or objective.

The normalized-geodesic T3 control at 258304 total transitions retained16left
and84right entries but only4right successes in100direct-policy episodes. Longer
gamma.999 and.99999 candidates both used only the right route at450k/500k/550k
and were stopped with logs and checkpoints preserved. A separate paired frozen
policy test increased the limit700→1400: right successes14→65, left0→1 out of100.
The extended-limit results are supplementary diagnostics, not native successes.
These findings do not establish the cause of collapse. They motivate checking
whether teacher candidate availability limits recovery of the less-used route.

Two fresh seed0 v3 jobs change only existing `actor.teacher_std_floor` from
exp(-5) to0.5 or1.0. The resolved `proposal_std` and `proposal_std_pretanh` retain
their existing interpolation. `ConditionalGaussianProposal` already applies the
same maximum(actor sigma, floor) to sampling and density correction. The new
adapter option forwards this parameter; it does not implement a new proposal.
All seven algorithm files pinned in register_horizon_temperature.py remain
byte-identical to2564b59faa0d319eece496b93eff0f19359efc37. The scratch probe uses
its own fixed key, verifies normalized bounded sampling/density and the default
floor identity, and never touches learner RNG, parameters or optimizer states.

Actor log sigma stays[-5,-1],initial-1; model256x3GELU, mean-init1, random latent,
N=M64, direct marginal NLL,beta1, density correction ON, teacherT3,gamma.999,
tau.005,actorLR3e-4/criticLR5e-4,Adam,replay1M,256env,batch4096,8updates/256,
warmup8192. DACER H/d+.7,interval500,initialalpha.27,alphaLR.03,noise scale.1,
GMM3/200; NovelD OFF. Normalized geodesic reward remains the archived control
profile:100*distance decrease,bonus0,step cost0. No reward implementation edit.

Budget250k post-warmup, actual258304total/7816learner updates by native counter.
Evaluate40episodes each50k and100final; preserve intermediate evaluation policies
and final full state/replay. v3 starts at its original fixed full state. Direct
policy=random latent+conditional sigma; mu-only separate; no DACER evaluation
noise. A candidate is not successful merely because it enters both corridors.
Compare successful route counts with failures included, teacher ESS, Q ranges
and same-step archived control. Widening proposals can worsen extrapolation or
slow learning; no automatic extension or success claim before measured results.

Commit/push/freeze before real8448-transition/batch4096 preflight and training.
Compare actual initial actor/critic hashes with the archived control and verify
the runtime configuration argument, plus periodic logged floor when available.
Run only two independently free180GPU slots,
preserving active v1/v4 schedule jobs and all locks. W&B OptiQ/antmaze,group
antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2. No4090 or unknown199 job resumes;
share committed source with199 after verifying its recovered connection and clean checkout. This is an explicit
ablation; omitted option and other benchmarks retain their existing defaults.

The4607dd5 first attempt completed8real preflight updates and the numerical
proposal check, then its new result validator wrongly required a periodic log
metric at step8448. The existing diagnostic interval omits that metric there;
this was not a learning failure. No main run started. Preserve the failed source,
checkpoints and logs. r2 uses a newly committed verifier of the runtime config
argument and always-logged actor_std_mean, with optional diagnostic comparison.
The algorithm, diagnostics interval and requested hyperparameters stay unchanged.
Register a fresh r2 campaign, never restart or overwrite the failed controller.
