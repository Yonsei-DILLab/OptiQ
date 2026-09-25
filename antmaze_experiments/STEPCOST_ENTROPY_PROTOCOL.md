# AntMaze v1 single-environment step-cost/XY-entropy comparison

Fresh seed0, temperatures .5/1/3/5/10, bonus coefficient0/.1: ten runs.
250000 total interactions includes10000 warmup;240000 updates,UTD1,batch256.
Unchanged basic DirectGMM/TRG256x2,N=M64,random z,logstd[-5,-1],DACER/NovelD off.
Extrinsic reward100*(nearest-goal Euclidean distance decrease)-1,no success bonus.
Original goal termination/time limits/training random XY reset unchanged.
Entropy ablation adds .1 times predictive -log p(nextXY) from cumulative .5m
grid occupancy on maze bounds with one pseudocount per cell (including walls).
Warmup visits count. This is a nonstationary XY novelty/occupancy-entropy proxy,
not full-state entropy, not differential entropy, and not an unbiased estimator.
Bonuses are stored with transitions and are NOT retrospectively recomputed.
Counts and per-transition bonuses saved with final checkpoint and verified against replay.
Every5000 including warmup and final250000:100 fixed-origin/full-pose/velocity
rollouts, fresh random z per action, conditional noise OFF,500-step limit.
NPZ coordinates/full starts/lengths/goals/returns plus PNG uploaded to jaehun-antmaze.
Evaluation uses extrinsic rewards ONLY. Evaluation interactions excluded from250K.
Intermediate evaluation-only policy checkpoints every5K; finalfullcheckpoint.
Four independently claiming supervised workers on idle vast1 GPUs; no old jobs resumed.
Each condition requires a real-update preflight before main; failures retained, no retry.
