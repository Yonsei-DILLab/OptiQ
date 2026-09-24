# v3 gamma .999 candidate: fresh 1M retention follow-up

Authorization: the active user goal permits bounded reward/hyperparameter
experiments and longer verification of promising candidates while forbidding
algorithm changes. The completed 250k screen is promising for entry diversity,
not yet successful: at258304 total transitions, direct policy n100 gave
left54/right33/uncommitted13 with zero goal successes, closest-goal mean4.716m.
Random-z mu-only gave left54/right20/uncommitted26, also zero successes.
Do not describe this evidence as multimodal goal-reaching.

Register one fresh OptiQ v3 seed0 policy on server180. Keep gamma.999,T1,
DACER H/d+.7, regulator interval500 and every learning/evaluation setting of
the gamma999 condition from eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45.
Use1M post-warmup transitions:1000192 actual post-warmup,1008384 total,
31256 learner updates and63 regulator updates. Evaluate40 episodes/mode every
50k total transitions and100 final. Save the same evaluation-only intermediate
policies and final full checkpoint. v3 uses original fixed full-state starts.
Compare direct-policy and random-z mu-only independently; never add external
DACER noise at evaluation. Check both successful routes at500k/750k/final1M.

This starts from seed0 again because the current runner has no checkpoint
resume interface. It is a longer same-seed follow-up, NOT an independent seed
or uninterrupted continuation. The short-run source/checkpoints remain intact.
No claim of independent replication is allowed. The current runtime hardcodes
seed0; do not label a later run seed1 without first implementing and validating
an actual seed override outside the algorithm core.

Reward100*(d_current-d_next), nearest-goal Euclidean, bonus0,step cost0,
NovelD OFF. Actor/twin critics256x3GELU, Adam actorLR3e-4/criticLR5e-4,
tau.005, random latent,N=M64,mean-init1,log sigma[-5,-1]/initial-1,beta1,
256env,batch4096,8updates/256transitions,warmup8192,replay1M are unchanged.
DACER initialalpha.27,alphaLR.03,noise_scale.1,GMM3/200samples are unchanged.
No actor/critic/TD/NLL/sampling/optimizer implementation or environment changes.
Registration verifies the seven core files against2564b59 and saves SHA256.

Commit/push/share this exact profile before real preflight and main training.
Use campaign antmaze-optiq-v3-gamma999-retention-1m-s0-20260925,
W&B OptiQ/antmaze. Server199 receives committed source only for this campaign.
Respect the existing horizon-screen pending jobs and GPU reservations; take an
otherwise free180 GPU with independent2second backfill. No existing learner is
stopped/restarted. Never use vast1/4090. Failure holds this queue and preserves
all artifacts; no automatic retry or hyperparameter change.
