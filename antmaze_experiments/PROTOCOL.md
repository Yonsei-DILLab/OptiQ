# Official DDiffPG sparse AntMaze:256 environments, four learners

Latest user approval2026-09-23: stop dense running/pending jobs and restart all16
policies fresh with original sparse reward, NovelD and256 parallel environments.
Methods OptiQ/SAC/DIPO/MFPO, v1-v4, seed0, eight GPUs. MEOW/DDiffPG itself are not
added. Preserve all cancelled source/logs/data; never resume old campaigns.

## Original source audit

Official https://github.com/supersglzc/ddiffpg HEAD was rechecked as
7edd06c4799abbab0f8fa534c21deb56253b018e. antmaze/ contains its159 unchanged files.
Check with python analysis_tools/verify_antmaze_upstream.py.

- ddiffpg/cfg/default.yaml: num_envs256, eval_num_envs20, sparse, NovelD,
  normalizeFalse, positional encodingTrue/L10, diffusion5steps/actionLR.03/20updates.
- ddiffpg/cfg/algo/{sac,dipo}_algo.yaml: batch4096,replay1M,horizon1,nstep1,
  update_times8,warm_up32,gamma.99,tau.05. Do not use the base actor_critic batch8192:
  the selected method configs override it with4096.
- Actor/critic AdamW LR3e-4/5e-4,max_grad_norm1,obs/value normalizationFalse,
  reward_scale1,handle_timeoutTrue. SAC automatic entropy uses alphaLR.005.
- DIPO critic DistributionalDoubleQ support[0,5],51atoms, restored exactly.
  Native5-step diffusion,20action-gradient steps, mixed exploration std.05-.6,
  target std.8 clipped to.2. No dense-specific support override remains.
- ddiffpg/utils/intrinsic.py: .01*max(n(next)-.5*n(current),0), RNDL2error,
  full29D observation with xy Fourier encoding, AdamW1e-4,clip1. Original imported
  code is used. Replay stores only sparse environment rewards; bonus is recomputed
  for each learner minibatch and RND is updated once per learner update.
- DDiffPG's own algorithm has warm_up500 and a trajectory/cluster replay. It is
  not among these four methods. We use the repo's SAC/DIPO baseline warmup32 and
  replay1M consistently for the four compared methods.

## Original environment

Use original MuJoCo2.1/mujoco_py environment and XML/maps; no physics port.
MuJoCo timestep.02, frame_skip5, RK4, actuator gear30, maze cell scale4,
29D qpos/qvel observation and8D actions in[-1,1]. No goal vector is appended
because original registrations set eval=True even for training; that flag also
makes goal arrival terminal. Base Ant locomotion rewards/fall-done are discarded
by the original GoalReachingEnv, not newly added here. Time limits remain.

| Maze | Goal xy | Goal reward | Horizon | Reset | Native max_step |
|---|---|---|---|---|---|
| v1 |(-8,0)|10|500|xy uniform[-2,2], original pose/zero velocity|3M|
| v2 |(-8,8);(8,0)|20;10|500|fixed original state|3M|
| v3 |(-12,12);(12,-12)|10;10|700|fixed original state|4M|
| v4 |(-16,4);(-16,-4)|10;10|700|fixed original state|5M|

Arrival radius<=.5; reward0 otherwise. v2's two goals deliberately retain the
original unequal payoffs. v1 uses low_gear_ant.xml; v2-v4 low_gear_ant_4g.xml.
preprocess_cfg changes v1 random_init toTrue even though default.yaml saysFalse.

## Shared collection/update accounting

All methods collect256 transitions then run8 learner/RND updates, batch4096,
replay1M. This is1/32 updates per transition and128 sampled replay items per new
transition. Warmup32 vector steps=8192 transitions, no learner updates.
Match baselines_main's native counter: global_steps excludes warmup and stop
when global_steps>max_step. Report actual total step and global_steps separately.

|Maze|Actual total transitions, including warmup|Learner/RND updates|
|---|---:|---:|
|v1/v2|3,008,256|93,752|
|v3|4,008,448|125,008|
|v4|5,008,384|156,256|

No invented1M cap. Each policy is one run with256 environments, not256 seeds.
SAC/DIPO import the original learners without model/LR/support changes.
OptiQ/MFPO are external adapters because upstream has neither; keep each native
model/objective/optimizer but match common data,batch,update and NovelD settings.
OptiQ:256x2GELU,T.25,beta1,DACERtrue,meaninit1,randomz,N=M64,
log_sigma[-5,-1]/initial-1,actor/criticLR3e-4; existing DACER state batch256.
MFPO: native256x3,2flow steps,native objective/optimizers/config.

## Evaluation, runtime and bookkeeping exceptions

Retain user-requested eval250k/final-only full checkpoint instead of upstream's
frequent eval/model saves. Interim20episodes/mode with20 vector slots, as original
eval_num_envs. Final100episodes per mode for natural and fixed-full-state resets.
Native: SAC mean, MFPO Q-best-of10, OptiQ randomz mu-only, DIPO native stochastic
reverse diffusion. Direct policy includes each model's policy noise, including
OptiQ conditional sigma. OptiQ zero_z separately. No NovelD or external Gaussian/
DACER exploration reward/noise during evaluation. Evaluation RNG isolated.

Seed0 is the user's single-seed comparison instead of upstream's generic seed42.
Python3.11/torchcu128 support5090; actual MuJoCo2.1/mujoco_py2.1.2.14/Gym.23.1
remain. A Gym/NumPy pickle compatibility adapter preserves RNG state exactly.
Explicitly seed the actual MuJoCo reset RNG (old Gym Env.seed is a no-op).
Preserve true terminal_observation before auto-reset for replay/NovelD; bootstrap
at time limits. These correctness/serialization adapters do not change reward,
physics, reset distribution or termination. No extra shaping or goal changes.

All256 training simulations are Gym AsyncVectorEnv CPU workers forked beforeCUDA
initialization, one GPU learner. Four jobs per server=1024 CPU simulations;
thread limits1, per-job CPU affinity. Do not claim256x speedup on fewerCPUcores.
Final full state: model/target/optimizer/entropy/DACER/RND/replay/RNG/256simulators,
SHA256/readback, sparse reward and nearest-goal success consistency validation.

## Validation and queue

Commit/freeze before any changed training. check_env audits256-worker v1-v4
physics/reward/reset/timeout, original resolved configs, native counter budgets,
replay wrap, and original NovelD formula. Each job also uses actual256env,
batch4096,warmup8192+256=8448 transitions,8real updates, finite changed model/RND,
save/readback and all evaluation modes before main. Preflights are not results.

Independent FIFO: immediately fill any finished GPU within2seconds, no all-maze
or all-method completion barrier. Respect GPU locks. Failure holds that server's
pending jobs, preserves live work; do not automatically restart failed training.
W&B OptiQ/gmm-trg, group antmaze-upstream-sparse256-nativebudget-s0-20260923.
