# OptiQ AntMaze: DDiffPG learning rates and 256x3 networks

Latest user approval (2026-09-23): stop existing experiments and launch fresh
OptiQ runs with the learning rates of the DDiffPG main method. The subsequent
instruction sets OptiQ AntMaze networks to 256x3. Apply this to BOTH actor and
twin critics, in the AntMaze adapter only. Do not alter other benchmarks.

The official vendored DDiffPG main configuration includes actor_critic.yaml:
actor_lr=3e-4, critic_lr=5e-4. Read these values directly from that file.
NovelD already uses its original RND predictor AdamW1e-4. RND is the novelty
estimator inside NovelD, not a separate additive RND reward. Compared with the
previous OptiQ run, critic LR changes3e-4 to5e-4 and hidden networks change
256x2 to256x3. Actor LR and RND LR are unchanged.

This is an LR/architecture change, not adoption of the entire DDiffPG method.
Keep OptiQ Adam, target tau.005, GELU, all other optimizer settings, T=.01,
beta1, DACERon, mean-init1, random normal latent, N=M64, log sigma[-5,-1]
with initial-1. Do not adopt the action-target optimizer LR.03, AdamW weight
decay, mode-Q, extra shaping, or the DDiffPG main method's longer warmup.

Retain the previous SPARSE_T001_PROTOCOL.md environment and evaluation profile:
original sparse reward10/20, NovelD.01,256env,batch4096,replay1M,warmup8192,
8 learner updates per256 interactions; seed0 each for v1-v4. Native total
budgets are3,008,256 /3,008,256 /4,008,448 /5,008,384 interactions. Original
training resets(v1random,v2-v4fixed) and random evaluation xy[-2,2] remain.
Every250k:40 native mu-only and40 direct-policy episodes plus an evaluation-only
policy checkpoint. Final100 episodes per mode/reset, zero_z separately, full
replay/model/RND/RNG/simulator save and SHA/readback checks.

Run four jobs on vast-heechan-180 GPUs0-3. Share the committed code with199;
do not register extra seeds/methods there. Old results and frozen sources stay
preserved. At approval-time inspection all previous four jobs had completed
and both servers had no active GPU processes; do not mislabel them cancelled.

Each job independently passes the existing real256env/batch4096/8-update
preflight before fresh main training. Also check actual actor/critic module
depths, parameter kernel shapes, and LR values using separate scratch Adam
states that do not mutate learner state or RNG. Original preflight parameter
change/RND/checkpoint readback and policy restore validations still apply.
No global completion barrier, automatic restart, or performance early stop.

W&B OptiQ/antmaze, campaign/group:
antmaze-optiq-ddiffpg-lr-256x3-s0-20260923-r3.
Register with antmaze_experiments.register_ddiffpg_lr_256x3 from its committed
frozen checkout. A failure holds pending work and preserves other live jobs.

The first registration (without-r2, source f07e4b2) stopped in preflight before
training: the shared MuJoCo config loader asserts its original256x2 profile.
Preserve that source and failed preflight logs. Apply AntMaze-specific model/LR
values after base-config validation and before model construction, then validate
the actual modules and optimizer transforms. The-r2 registration is a fresh
preflight/main launch with the same approved experiment settings, not a resume.

The-r2 preflight (source297aad7) constructed the requested model but stopped
before learning because the added verifier referenced a nonexistent policy.actor
attribute. The-r3 verifier checks actual initialized hidden-layer kernel shapes
(including the twin-critic axis), and retains the independent optimizer probe.
Both failed attempts remain preserved; no main training ran in those attempts.
