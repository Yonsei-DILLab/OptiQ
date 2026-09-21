# Multi-goal and multi-path policy comparison

## Objective and primary endpoint

Train each policy online, then repeat stochastic rollouts of that ONE policy.
Measure whether it discovers and retains several successful goals/routes or
concentrates on one behavior. Never pool different training seeds into one
apparently diverse policy. First analysis is at100,000 ENVIRONMENT interactions,
as in MFPO Figure10. This campaign does not automatically extend to1M.

Defaults: DDiffPG v1 andv4, OptiQ/SAC/MEOW/SQL/MFPO/DIPO, seed0–3,48 runs.
The implementation supports v3 if the user requests all three layouts.
Server199 GPUs0–3 are exclusively assigned to this comparison. Each available
GPU backfills immediately; no environment completion barrier. Stop pending
launches on failure and preserve other running jobs. No automatic retries.

## Source evidence and port

- MFPO E.6/Figure10: https://arxiv.org/pdf/2604.14698#page=24.
  Its caption identifies100k interactions; its public repository atd8b3977
  has no AntMaze task or reward implementation. This is a reference experiment,
  not an exact replication of an unavailable MFPO AntMaze configuration.
- MaxEntDP AppendixD.2: https://arxiv.org/html/2502.11612v3#A4.SS2 explicitly
  describes a dense penalty for distance to the closest goal on DDiffPG mazes.
  Exact shaping scale, termination, and implementation are not specified there;
  the inspected public repository at8adfc7e does not contain this environment.
- DDiffPG: https://github.com/supersglzc/ddiffpg/tree/7edd06c4799abbab0f8fa534c21deb56253b018e.
  Maps extracted from maze_env.py; original low_gear_ant.xml is vendored with
  licensing/provenance hashes. Scale4, walls half-size(2,2,1), frame_skip5,
  physics timestep.02, RK4, motor gear30. Goals: v1(-8,0),
  v3(-12,12)/(12,-12), v4(-16,4)/(-16,-4). Start(0,0).
  v1 horizon500; v3/v4 horizon700. Observation29D=qpos15+qvel14,
  action8D[-1,1], no individual goal conditioning or contact observations.

We port the robot/geometry/reset contract from mujoco_py to MuJoCo3.3.7 and
Gymnasium1.2.3. Legacy compiler coordinate=local is removed because local is
the modern default. Simulator versions differ; bitwise legacy dynamics are
not claimed. No robot geometry, gear, integrator or timestep is retuned.

Reward is explicitly r=-min_g ||next_xy-g||2, scale1, no added success bonus,
locomotion, control-cost or survival reward. Any goal within.5 terminates.
Timeouts bootstrap. The original DDiffPG goal wrapper discards Ant fall
termination; this behavior is retained. This dense reward replaces the upstream
sparse success bonus. All algorithms receive exactly the same environment.

Reset follows DDiffPG preprocess_cfg: v1 xy uniform[-2,2]^2, v3/v4 fixed origin;
robot default model pose and zero velocity. No selected goal appended to state.
An environment contract test checks geometry, dense reward, all goals,
terminal vs time-limit behavior, identical full simulator states and route labels.

## Learners

Existing native online implementations from antmaze/agents.py are reused.
OptiQ T=.25,beta1,DACERtrue,mean-head variance scale1,random latent,N=M64,
256x2 GELU,log sigma[-5,-1],initial-1,actor/critic Adam3e-4,batch256,UTD1,
warmup5k,unclipped gradients. DACER Ant noise_scale.1,Htarget=-.9*action_dim,
alpha.27,alpha_lr.03,interval10k,GMM3,200samples/state,behavior-only.
SAC/SQL warmup5k,256x2,LR3e-4; DIPO native100-step diffusion,
20action-improvement iterations,LR3e-4; MEOW continuous-control native
64x2 flow/256x2 scale,LR1e-3,alpha.2; MFPO native config256x3,T2,LR3e-4,
initialtemp.01,16/32proposals,C51[-1600,1600]/101atoms,10k warmup.
All batch256/UTD1. Native baseline architecture/optimizer differences are
disclosed, not silently homogenized. Equal interactions are not equal FLOPs.

## Evaluation: stochastic behavior of a learned policy

Every5k:10 direct-policy rollouts, independent evaluation RNG.
Final100k:100 natural-reset rollouts AND100 identical-full-state rollouts
per policy. Physics integration state (including warm-start/internal state) is
saved and equality asserted for fixed-start rollouts. V3/v4 native reset is
already deterministic; those two reset protocols do not provide independent
initial-state conditions and must not be pooled as200 independent observations.

SAC samples its Gaussian; MEOW samples its flow; MFPO samples its MeanFlow
without best-of10 selection; DIPO samples its stochastic diffusion generator;
SQL samples its learned generator. OptiQ's primary draws include random latent
and learned conditional sigma, excluding additional external DACER exploration
noise. Random-latent mu-only output is a separate supplemental analysis.
Zero-latent/mean actions and best-of-K selection are not the primary comparison.

Store every trajectory, length, return, reached goal, environment/policy seeds,
initial observation and complete simulator state. Periodic eval visits never
enter training coverage. Training xy occupancy uses.5-unit bins.

Per training seed report: success/failure rate; goal reach fractions (over all
episodes); route-use fractions; dominant successful route share; entropy and
effective goal/route count; successful path length; training coverage. Zero
success means zero observed successful modes, not a learned deterministic mode.
Unknown route assignments are reported separately and excluded from mode counts.

Route labels are PREDECLARED GEOMETRIC GATES, not proof of action-distribution
multimodality or a complete mathematical homotopy classification. V1 uses the
last leftward crossing ofx=-4 (upper/lower). V4 uses firstx=-4 crossing
(upper/lower entrance) and lastx=-12 crossing (upper outer/middle/lower outer),
plus reached goal. Unexpected long routes remain visible. These labels separate
the canonical feasible paths without treating minor trajectory jitter as modes.

## Reproducibility and completion

Commit before training; freeze source and all native submodule versions; record
full SHA, XML SHA256, package lock, configs, W&B URLs. W&B OptiQ/gmm-trg,
group antmaze-multimodal-100k-20260921-<task>. Independent supervised controller.
Preflight uses272envsteps,256warmup,2episodes,disabled W&B and separate dirs;
it checks changed finite actor/critic parameters, counters, trajectories,
full-state identity and checkpoint hashes. It is never included in results.
Real100k update audit:95k for5k warmup methods,90k forMFPO. Checkpoints50k/100k.
Completion includes all requested seeds, raw rollouts, per-seed figures,
aggregate mean±sampleSD tables, source/config provenance and local backups.
