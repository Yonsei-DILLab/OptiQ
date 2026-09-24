# v3 normalized reward: discount and later acquisition

The active goal permits bounded reward/hyperparameter screens with the OptiQ
algorithm unchanged. The completed normalized-geodesic T3 run retained left16
and right84 entries among100 direct-policy episodes at258304transitions, but
only4 right successes. Mu-only retained left8/right92 with38 right successes.
There is no left success or successful multi-route result yet. The T1
normalized candidate had no left entries and is not extended.

Test two fresh v3 seed0 policies with the exact same normalized reward and T3:
gamma .999 (longer-budget control) and .99999 (discount-only treatment).
The250k T3 source is db4ca0a446f5fe8df1dda02e261fa9154c66d9c9.
Keep the same reward potential, including fixed reference weights and distance
to the existing radius-.5 success regions. No new reward terms, direction
labels, episode-history state, bonus, time cost or NovelD are introduced.

Read-only rescoring used74 successful left trajectories from a historical
Euclidean policy and4 successful right trajectories from the normalized T3
policy. Under the normalized reward their mean returns are1447.70/1326.34 at
gamma.999,1644.86/1643.23 at.99999, and1647.0563/1647.0563 without discount.
These are different historical policies and budgets, conditioned on success.
They establish properties of the reward functional, not causal evidence that
changing gamma repairs the critic or a report of newly successful policies.
The original failures and complete denominators remain preserved.

With a zero potential throughout both success regions, undiscounted progress
rewards telescope to the same start potential on both successful paths. At
gamma<1 the timing of progress still matters. This screen reduces that timing
preference. Near-unit gamma can also slow or destabilize TD learning, and does
not force equal learned Q values or successful routes. Gamma1 is only an
analytic reference and is not used for training.

Only the budget/identity change for the .999 control; .99999 additionally
changes the existing discount hyperparameter. Keep T3, H/d+.7,500update DACER
interval,alpha.27,alphaLR.03,noise scale.1,GMM3/200,256x3 actor/twin critics,
LR3e-4/5e-4,Adam,tau.005,random latent,N=M64,beta1,mean-init1,
log sigma[-5,-1]/initial-1,256env,batch4096,8updates/256,warmup8192,replay1M.
No core algorithm, runner, reward implementation, sampler, optimizer, TD,
NLL, maze physics, reset or terminal code changes. Core7files must match2564b59.

Maximum1M post-warmup:1008384total transitions,1000192post-warmup,
31256learner updates,63DACER updates. Every50k retain40episodes per mode and
evaluation-only policies; final100episodes plus full state. Use the original
v3 fixed full starting state. Direct policy retains random latent and
conditional sigma; mu-only is separate. No external DACER noise at evaluation.

Screen250k/500k/750k. Successful distinct routes, not entry counts, are the
positive criterion. If the minority route remains absent at two later
checkpoints with no minority success, preserve artifacts and stop that
candidate early. No automatic extension, retry or mixing checkpoints/seeds.
The .999 run is a fresh same-seed longer-budget control, not a checkpoint
resume or an independent replication; verify shared intermediate prefixes.

Commit/push/share/freeze before each real8448transition/8batch4096update
preflight. Check source/config, identical initial parameters and full
checkpoint/replay readback. Run only on180 with supervisor and existingGPU
locks, preserving live v4T3/T10 jobs. Independent backfill; no all-job barrier.
199 source sharing remains pending SSH recovery. Do not restart unknown199
jobs, cancelled campaigns or use4090. W&B OptiQ/antmaze, separate group.
