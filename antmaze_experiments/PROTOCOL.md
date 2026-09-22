# Official AntMaze, 64 environments, four learners

User approval: 2026-09-23. Stop old v1 DIPO. Fresh seed0 v1/v2/v3/v4,
OptiQ/SAC/DIPO/MFPO, 16 policies, eight GPUs. The user explicitly selected
64 environments per policy and batch4096 for all four methods.

The complete unmodified DDiffPG source is in `antmaze/` at upstream commit
7edd06c4799abbab0f8fa534c21deb56253b018e. These adapters are outside that tree.
Use actual MuJoCo2.1/mujoco-py2.1.2.14/Gym0.23.1, no modern-physics port.
Original map/low-gear XML/29D observation/termination/sparse reward0,10,20 remain.
v1 uses random_init=True, as upstream preprocess_cfg; v2-v4 use False.
NovelD is imported from upstream: .01*max(n(next)-.5*n(current),0), no normalization,
full observation with xy positional encoding10 bands, original RND/AdamW1e-4/clip1.

## Equal data and update budgets

All methods: batch4096, replay1M, 64 simultaneous CPU environments, one GPU learner.
Collect64 transitions and perform2 actor/critic/RND updates. This preserves original
DDiffPG 256-env/8-update ratio: 1/32 optimizer updates per collected transition,
128 replay samples drawn per new transition after warmup. Both OptiQ and MFPO now
use this common ratio, not their old UTD1/batch256 settings. Model architecture,
learning rates, objectives, entropy and target-update rules remain method-native.

Warmup8192 transitions for all, equal to original32 vector steps times256 environments.
The total budget includes warmup: exactly1,000,000 transitions,15,625 vector steps,
30,994 learner and RND updates. Each vector environment contributes15,625 transitions.
Unlike the original runner, warmup is counted and the loop does not overshoot1M.

SAC/DIPO use original AgentSAC/AgentDIPO and native batch4096 configs, with
update_times=2. SAC/DIPO actor/critic AdamW3e-4/5e-4,tau.05,gamma.99; original
network sizes retained. DIPO uses original5-step diffusion and20 action-gradient
steps, distributional critic, native mixed exploration noise. It is not the old
100-step external DIPO implementation. SAC uses native automatic entropy tuning.
OptiQ uses TRG direct-GMM,256x2GELU,T.25,beta1,random latent,N=M64,log_sigma[-5,-1],
initial-1,mean-head scale1,DACER=true,noise_scale.1,target_entropy_per_dim-.9,
alpha.27,alpha_lr.03,10k learner-update interval. Diagnostic state count stays256.
MFPO uses native256x3,2 flow steps,native objective/optimizers/config.

## Collection and evaluation

Gym AsyncVectorEnv runs64 MuJoCo CPU workers. Policies draw independent batched
randomness. Thread counts1 and per-GPU CPU affinity limit oversubscription.
The two servers have different CPU quotas; measure actual throughput, do not
claim64x acceleration. Env workers are forked before CUDA initializes.
The bookkeeping wrapper preserves terminal_observation before Gym auto-reset,
bootstraps across time limits, and records true terminal positions for NovelD.
No extra shaping or reset/success/goal modification is added.

Evaluation every250k threshold, after the current64-step batch:250048,500032,
750016,1000000. Interim native/direct-policy10 episodes; final100 per mode for
natural and identical full-state starts. OptiQ also gets zero_z. Rollouts retain
failures and per-policy identity. Direct policy includes SAC policy noise, DIPO
native diffusion noise, MFPO latent noise, OptiQ latent and conditional sigma.
Native: SAC mean, MFPO Q-best-of10, OptiQ random-z mu-only; upstream DIPO native
diffusion is itself stochastic. No external Gaussian/DACER exploration or NovelD
reward in evaluation. Evaluation RNG is isolated from training. v2-v4 natural
starts can already be identical; do not pool natural/fixed into200 distinct starts.

Only final full checkpoint: models/targets/optimizers/entropy/DACER/RND/replay,
RNGs and64 simulator states, SHA256 and serialized read-back verification.
The preflight uses the same actual batch4096,64 environments,warmup8192 and
4 real learner updates to8320 total transitions; it is not a reported result.
Require finite changed actor/critic/RND predictor and unchanged RND target,
exact update counts, successful checkpoint readback and all evaluation modes.

## Scheduling

Commit exact code and freeze before preflight or main training. Each GPU job
runs its own preflight then main training; no maze/method completion barrier.
Each completion opens the slot to the next job within2 seconds. Respect GPU locks.
Failure holds new jobs on that server and preserves other live jobs; no automatic
restarts. Preserve previous source/logs/checkpoints and cancelled queues.
W&B OptiQ/gmm-trg, group antmaze-upstream-64env-1m-s0-20260923.
