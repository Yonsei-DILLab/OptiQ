# Requested25-environment UTD1 comparison

Four fresh iBOLT v1-v4 seed0,T3 runs on idle vast2 RTX5090 GPUs0-3.
Same dense nearest-goal negative distance and basic algorithm as single-env
controls:256x2,N=M64,beta1,log_std[-5,-1],gamma.99,LR3e-4,DACER/NovelD OFF.
Batch256,25 parallel simulators,each vector step produces25transitions and
is followed by25 sequential gradient updates from independently sampled batches.
Warmup10,000 total transitions=400vector steps,no updates during warmup.
Total1,000,000 transitions=40,000vector steps;39,600learning blocks give
990,000learner updates. This is NOT one update per vector step and NOT25Mdata.
Preflight10,050transitions validates2learning blocks/50updates,25simulators,
parameter changes,replay/env serialization,original reward,finite losses.
Fixed-origin eval every25K,100episodes each mu-only random-z/full policy;
intermediate policy checkpoints,final full state. Preserve existing single-env
controls and all frozen source. Only stop explicitly requested MaxEntDP T.1/.5/1.
W&B OptiQ/jaehun-antmaze group ibolt-antmaze-t3-env25-utd1-20260926.
