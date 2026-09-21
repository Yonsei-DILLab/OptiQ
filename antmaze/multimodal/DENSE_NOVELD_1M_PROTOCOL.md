# AntMaze dense + NovelD, 16 runs

User-authorized scope: OptiQ, SAC, MEOW, MFPO (the user's MFPR spelling is
interpreted as MFPO), DDiffPG mazes v1/v2/v3/v4, seed 0, 1,000,000 environment
interactions per run. Two hosts run four exclusive GPU jobs each, backfilling
every two seconds. No old canceled jobs are resumed. Failed jobs hold the pending
queue and preserve other running jobs; no automatic restart.

Environment: vendored DDiffPG 7edd06c geometry/low-gear Ant, modern MuJoCo port,
29-D qpos+qvel, 8-D bounded actions, gear 30, timestep .02, frame skip 5.
v1/v2 horizon 500, v3/v4 700, goal radius .5, goal termination, timeout bootstrap.
v1 random xy in [-2,2]^2; other maps fixed origin. v2 adds upstream MAZE_v2,
goals (8,0),(-8,8). No locomotion, success, curriculum or route bonuses.

Dense environment reward is minus Euclidean distance to the closest goal after
each transition, as described in MaxEntDP D.2. MaxEntDP's AntMaze-specific code
is unpublished; this is not a claim of exact reproduction. NovelD is a user-
requested extension from DDiffPG: .01 max(n(s')-.5 n(s),0), RND L2 error,
no normalization, xy encoding 10 bands, fixed target, AdamW1e-4/clip1 predictor.
Recompute bonus at replay sampling, update predictor once per learner update.
Store only environment reward in replay; evaluation never includes NovelD.

One environment, batch 256, UTD 1, replay capacity 1M. Native agent networks,
optimizers, learning rates, discount, target updates and warmup are preserved.
OptiQ: T=.25, beta1, DACER true, mean-init1, log sigma[-5,-1], initial-1,
random z, N=M64, 256x2 GELU, LR3e-4. Native DACER behavior noise only in training.
SAC/MEOW/OptiQ warmup5k; MFPO10k. NovelD does not modify the policy entropy term.
Do not use the older 256-env/batch4096/UTD1/32 NovelD pilot profile.

Every 25k: 10 natural-reset episodes each for native and direct stochastic
evaluation. Native SAC=tanh(mu), MFPO=Q-best-of10, MEOW=center prior,
OptiQ=random-z mu-only. Direct policy includes native distribution noise
(including OptiQ conditional sigma); no added DACER exploration noise.
At 1M: 100 episodes per evaluation mode and reset distribution, both natural
and identical full simulator state; OptiQ zero-z mu-only additionally. Save raw
xy trajectories, success, goal/route occupancy and training state coverage.
v2 route metric measures goal choice through the central corridor, not invented
multiple paths to one goal. Do not pool multiple policies into a diverse policy.

Every 100k and at completion: atomic resume snapshots contain all replay rows,
ring index, all model/target parameters and optimizer states, learned entropy,
DACER optimizer/noise RNG, NovelD target/predictor/optimizer, Python/NumPy/Torch/
JAX RNG, environment simulator integration state, environment RNG, episode and
training counters, coverage and evaluation histories. Each file has SHA256.
Keep all checkpoints, including final 1M, to permit later extension.
CLI --resume <snapshot> --steps <higher-total> retains source SHA and warmup,
restores rather than recreates replay, and writes a separate output directory.

Before production: physics/dense reward/goal tests for all maps, followed by
fresh-process 272-step smoke (256 warmup,16 updates), load its full checkpoint
and continue to280 (24 total updates), for every method/map. Check every state
digest before continuation, exact replay, targets/RNG, parameter changes,
RND fixed target and learned predictor, evaluation and raw rollout consistency.
Smoke runs use disabled W&B and are excluded from the 16 production runs.
Commit exact source before any preflight or production launch.
