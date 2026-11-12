# Pure online Adroit Sparse-v2: SAC and TD3

User authorization, 2026-11-12: run seeds **4 and 5**, vary only vector environment
count and UTD, start with 256 environments, use four GPUs on the supplied server.
The user then expanded the allocation to **eight GPUs across two servers**.
Use branch **heejoon-rebuttal**. This supersedes the earlier seed5/6 pilot and the
workspace's older branch instruction for this study only. No iBOLT, alpha/noise
tuning, or automatic confirmation on seeds0-3 is authorized in this campaign.

Reference: heechan-rebuttal ebdc46e59db1642902e0331c1e3aa662556c6553.
The existing A1 demonstration-based experiment is not resumed or modified.
The active servers are heejoonorm@175.155.64.205:19159 (shard0) and
heejoonorm@179.255.127.236:31020 (shard1). Both expose four RTX5090 GPUs.
Each server receives all four tasks and both algorithms; shard0 has SAC seed4
and TD3 seed5, shard1 has SAC seed5 and TD3 seed4. All profiles keep this mapping.
The originally supplied
203.75.167.56:42913 failed device access and has no training jobs from this study.

## Fixed scientific settings

- Door, Hammer, Pen, Relocate: native `AdroitHand*Sparse-v2`, horizon200,
  upstream Gymnasium-Robotics b8da02a89191daf8f004e7f7510ff655f7a90100.
  Reward is +10 on native success and -0.1 otherwise. Preserve resets/physics.
- Randomly initialized actor/critics and empty standard online replay only.
  No demonstrations, checkpoints, BC, HER, shaping, curricula or normalization.
- SAC: unmodified shared `baselines/jax_sac.py:JaxSAC`, actual JAX GPU backend,
  auto-alpha and target entropy minus action dimension. SB3 reference initialization.
- TD3: unmodified SB3 2.7.0 learner, exploration Gaussian std0.1 in [-1,1]
  coordinates, policy delay2, target noise0.2 clipped at0.5; deterministic evaluation.
- Both: actor/critic256x2 ReLU, twin critics, LayerNorm off, lr3e-4,
  gamma0.99, tau0.005, minibatch256, replay1M, four CPU collection workers/run.
- Requested3M interactions. Use **3,000,064** for every profile (next multiple
  of256, excess64=0.00213%). Warmup **10,240** (10k rounded to256) is identical
  across profiles. Evaluations do not count toward training interactions.
- Only profiles differ: (env256,UTD0.25), (env256,UTD1), (env64,UTD1).
  Each has2methods x4tasks x2seeds=16 runs;48 full runs in total.
  Priority order follows this list, but independent tasks have no afterok gates.
- Collect one vector step, then n_envs*UTD critic updates. Twin-Q optimization
  counts as one update; TD3 actor updates every other critic update.
  Expected critic updates:747,456 or2,989,824.

## Evaluation and provenance

- Initial policy and a separate uniformly random-action diagnostic, each100
  episodes. Regular evaluations every50,176 transitions use100 episodes;
  final evaluation at3,000,064 uses1000 episodes. SAC deterministic primary,
  stochastic secondary; TD3 deterministic primary. Fixed reset schedules are
  paired by training seed across algorithms/profiles and milestones.
- Evaluate after the update block. Evaluation preserves Python/NumPy/Torch/JAX
  learning RNG and never writes replay. Report any-step native success, final-step
  success, success step count and >=25 success steps (not necessarily consecutive).
- Record first online success as vector-step-end transition count and environment
  index, warmup successes, all episode success counts and wall times. Initial/random
  successes and isolated training successes do not establish learned competence.
- CPU simulator checks verify rewards, shapes, seed determinism and native timeouts.
  GPU preflights verify real learner updates, timeout bootstrapping, evaluation RNG,
  finite parameters and exact update counts before full runs. Smoke results are
  separate, not part of the benchmark. No CPU fallback for reported learning.
- Commit source/configuration/protocol/launch scripts before smoke or full runs.
  Deploy an immutable hash manifest with full source SHA, branch, runtime versions,
  run IDs, GPU allocation and process IDs. Do not claim bit-exact resume: saved
  policies, optimizers and final replay support analysis but environment/RNG state
  is not fully resumed by this launcher. Failed runs retain outputs, never overwrite.
- Use Supervisor for the controller, with a GPU memory admission guard and
  staggered compiler starts. Start at up to2 jobs/GPU if memory/CPU checks permit;
  adjust operational concurrency from measured throughput without scientific edits.
- Store results/checkpoints outside Git; archive to
  dildata:/data1/heejoonorm/OptiQ/runs/adroit-online-20261112.
  Retain inference checkpoints every250,880 and final; final replay is saved.

## Execution

The deployment script verifies clean tracked inputs, creates a git-archive source
snapshot and manifest, and transfers it to the explicitly supplied server. Runtime
installation reuses the pinned Adroit requirements. The controller requires an
explicit GPU-preflight receipt tied to the source SHA; it refuses unverified starts.
Never start jobs from mutable working copies. The monitor produces JSON/CSV summaries
with current steps, first success, evaluation success and throughput per run.
