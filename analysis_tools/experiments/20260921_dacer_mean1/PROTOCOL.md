# DACER with Xavier mean-head initialization, 2026-09-21

User authorization: run HalfCheetah seeds 0,1 and Ant seeds 0,1,2,3 with
mean_output_init_scale=1.0, temperature=0.25 and DACER enabled on the four idle
RTX 5090 GPUs of vast-heechan-199. There are six runs: launch HalfCheetah 0,1
and Ant 0,1 first, then Ant 2,3 on whichever GPU becomes free first.

Branch: direct-gmm-trg. Commit the exact profile, plan, controller and protocol
before launch. Run from a clean frozen clone named by the full commit SHA.
Record the SHA and resolved configuration in every job. No changes to existing
frozen sources, old results, the GMM40 queue on vast-heechan-180, or shared RL
defaults. The mean-head override belongs to this experiment profile only.

One million environment steps per run. Actor and twin scalar critic: 256x2
GELU; random normal latent, N=M=64, conditional box-truncated Gaussian mixture,
direct marginal GMM NLL; beta=1 with density correction. Log sigma bounds
[-5,-1], initial log sigma=-1, log-sigma head kernel scale=0. Mean-head kernel
variance scale=1 with fan_avg/uniform, exactly Xavier uniform. Batch 256, UTD1,
actor/critic Adam LR3e-4, replay 1M, warmup5000, gamma.99, tau.005, no gradient
clipping. Every algorithm setting is asserted equal to the existing TRG base
except the requested mean-output initialization scale.

DACER defaults unchanged: target entropy=-.9*action_dim, initial alpha=.27,
alpha LR=.03, interval10000 learner updates (including first update), fitted
GMM3 components and200 policy samples per replay state. Existing environment
noise scales: HalfCheetah .15, Ant .1. DACER affects behavior actions only.
Teacher, TD target and evaluations have no extra DACER noise.

Evaluation every5000 steps,10 episodes each for zero_z and stochastic_z.
Both are mu-only; stochastic_z uses fresh normal latent per action.
Primary comparison: stochastic_z mean over900000<step<=1000000, exactly
20 evaluations x10 episodes per seed, then seed mean and sample standard
deviation. zero_z is reported separately. No performance-based early stopping.

W&B: OptiQ/gmm-trg. Groups trg-dacer-mean1-20260921-<task>-T0.25-b1.
Names <task>-trg-dacer-mean1-T0.25-b1-s<seed>.
Campaign: /home/heechan/optiq-experiments/trg-dacer-mean1-20260921.
Reuse the pinned sklearn dependencies from the previous DACER campaign read-only.
Supervisor service: trg-dacer-mean1-20260921, four GPU slots with shared OptiQ
GPU locks and a two-second backfill loop. No barrier between environments.
No automatic retries: a failure blocks pending jobs while active jobs finish.
Completion requires config/source identity, final actor/critic checkpoints,
and all finite final-window evaluations. Preserve any final W&B upload warning.
No beta sweep, additional seeds, or new environments are authorized by this plan.
