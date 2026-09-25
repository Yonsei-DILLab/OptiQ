# Five-environment T5 replacement

User confirmed replacement of only vast4 single-env T3 v1-v4 jobs.
Preserve logs/checkpoints and all vast2 env25 T3 jobs.
Four fresh seed0 v1-v4 runs: T5, 5 simulators, batch256, 5 independently
sampled sequential learner updates after each 5-transition collection block.
Warmup10000 total transitions; budget1M total; expected990000 learner updates.
Preflight10010 transitions checks10 updates and5 serialized simulators.
Preserve basic DirectGMM/TRG: N=M64,beta1,256x2,logstd[-5,-1],LR3e-4,
gamma.99,ordinary TD,DACER/NovelD off,negative nearest-goal distance reward.
Fixed-origin evaluation every25K:100 episodes each random-z mu-only and full
policy. Upload mu-only trajectory PNG alongside scalar metrics and raw NPZ.
Final evaluation also zero-z. Keep intermediate evaluation checkpoints.
W&B OptiQ/jaehun-antmaze, group ibolt-antmaze-t5-env5-utd1-20260926.
vast4 GPU0/1/2/3 map to v1/v2/v3/v4. Frozen committed source before launch.
