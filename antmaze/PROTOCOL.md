# AntMaze online comparison

The user assigned vast-heechan-199 GPUs 0–3 after stopping two HalfCheetah and
two Ant runs. Pending Ant seeds 2–3 remain deferred. No old run is resumed.

## Task and budget

Official Gymnasium-Robotics `AntMaze_UMaze-v5`, sparse reward, continuing_task=False,
1,000-step episode limit. Default robot contact observations are retained.
Flatten observation, achieved_goal, desired_goal in that explicit order: 109 dimensions;
8 actions in [-1,1]. No HER, offline data, reward shaping or observation normalization.
Success is official info['success']; reaching goal terminates; timeouts bootstrap.
All six algorithms run seeds 0,1,2,3 for 1,000,000 environment steps each.
Batch=256, one learner update per post-warmup environment step. No early stopping.
Equal steps do not mean equal FLOPs, Q queries or network sizes.

## Algorithms

- OptiQ: existing Direct box-truncated GMM/TRG online implementation and behavior-only
  DACER regulator. T=.25, beta=1, mean_output_init_scale=1.0, log sigma [-5,-1],
  initial -1, normal fresh latent, N=M=64, 256x2 GELU, Adam 3e-4 actor/critic,
  warmup5k, UTD1, no gradient clipping. DACER Ant noise_scale=.1, Htarget=-.9*8,
  initial alpha=.27, alpha_lr=.03, interval10k, fittedGMM3, 200samples/state.
- SAC: SB3 online twin-Q SAC, 256x2 ReLU, Adam3e-4, warmup5k, automatic entropy.
- DIPO: upstream c6d8d1b, native diffusion/critic networks, 100 diffusion steps,
  action improvement20, diffusion/critic LR3e-4, actionLR.03, native clipping2,
  warmup5k. Existing correction to advanced-index replay writeback retained.
- MEOW: upstream b786d27 continuous-control FlowPolicy and Bellman loss, not toy2D.
  Native 64x2 additive coupling networks / 256x2 scaling, alpha.2, LR1e-3,
  gradclip30, log sigma[-5,-.3], warmup5k. No separate SAC-style policy loss.
- MFPO: upstream d8b3977 configs/mfpo_config.py unchanged; 256x3, layernorm,
  LR3e-4, initialtemp.01, T2, proposal16/32, C51[-1600,1600]/101atoms,
  native 10k warmup, evaluation10candidate Q-selection.
- SQL: repository JAX port of haarnoja/softqlearning 6f51eac; dynamic state/action
  dimensions, learned soft Q, original SVGD and TF1-compatible Adam, 256x2,
  LR3e-4, T1,16kernel/value particles, hard target1000 envsteps, warmup5k.

Native baseline hyperparameters are retained; in particular MEOW LR/architecture
and MFPO warmup/architecture differ. Environment and evaluation budgets are shared.

## Evaluation and outputs

Step0 and every5k:10 episodes, paired reset seeds across algorithms for each seed/step.
Evaluation RNG is isolated from learning. OptiQ primary=stochastic_z conditional
mean (fresh z every action), auxiliary=zero_z; neither adds conditional sigma or
DACER noise. Other methods use native evaluation (SAC deterministic mean,
DIPO eval sampler, MEOW deterministic, MFPO candidate selection, SQL generator).
Report success rate, return, episode length, final/minimum goal distance. Sparse
episodic return equals 0/1 success. Last100k is 900000<step<=1000000:
20 evaluations x10 episodes, per-seed average then four-seed mean and sample SD.
Trajectory for episode0 at step0/500k/1M. Save NPZ histories and checkpoints100k.
Partial/failed seeds are never treated as completed final comparisons.

Each job records exact committed source SHA, package versions, algorithm config,
W&B URL, progress, final checkpoint SHA256, update audit and result.json.
W&B: OptiQ/gmm-trg, group antmaze-umaze-online-20260921.
Preflight --smoke uses272 steps/256warmup/2evalepisodes, W&B disabled, separate
output directories; it is excluded from all results.

## Environment isolation and queue

/home/heechan/.venv-optiq-antmaze shadows only Gymnasium1.2.3,
Gymnasium-Robotics1.4.2, MuJoCo3.3.7, SB3 2.7.1 and PettingZoo1.24.3;
read-only fallback .pth uses the existing gmm40 runtime (JAX.4.33/Flax.9,
Torch2.7.1+cu128/NumPy1.26.4). It never installs into the existing runtime.
Full resolved package lock is captured with each campaign; pip check passes.
Learners use the existing run-gpu.sh exclusive GPU locks. A supervisor controller
backfills any free slot every2s without stage barriers. A failed job freezes
pending launches but preserves other live jobs; there is no automatic restart.
