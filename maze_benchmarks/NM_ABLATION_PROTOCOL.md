# PointMaze independent N/M ablation — 2026-09-26

User requests seed0, T5, N in {1,4,16,128,256} with M64 and the reverse
M in {1,4,16,128,256} with N64. Simple/Medium/Hard: 30 fresh runs.
Reuse completed N64/M64/T5 controls; no control reruns.

Keep source pointmaze c5cb1fa, original DrAC sparse +100 maps,256env,
batch4096,16updates/256transitions,8192warmup,1000192total,62000updates,
256x2 networks,log_sigma[-5,-1]/init-1,beta1,ordinaryTD,DACER/NovelD off.
Primary and obstacle evaluation: fresh-z mu-only,200episodes/200K and500final.
Save raw rollouts, automatic trajectory PNG, checkpoint and final replay.
W&B OptiQ/pointmaze-NM-ablation. Persist training metrics every1000updates.

Only algorithm extension: independently specify the number M of iid exact
mixture candidates. Sample component IDs uniformly WITH replacement from N
conditionals, evaluate their density under those SAME N components, retain
the existing direct marginal NLL. No OT/resampling/stratification introduced.
Omitted M preserves legacy N*repeats behavior. Explicit N64/M64 must match it
exactly under a fixed RNG. M1 has unit normalized weight: value weighting
cannot influence that actor update. N1 has one Gaussian in each sampled
state-specific mixture, not a globally fixed z or a deterministic policy.

Retain existing N>=256 actor microbatch256 implementation (full batch4096,
one mean gradient / Adam step; PRNG partition differs). Do not silently
reduce batch size, N, M, environment count or UTD after OOM.
Run real8448transition/16update preflight before each independent main run.
Verify actual shape metrics, unchanged settings,finite updates, raw eval and
checkpoint accounting. Preserve failures without retry. Use idle GPUs only,
common GPU locks and no preemption. Commit/push/freeze exact source first.
