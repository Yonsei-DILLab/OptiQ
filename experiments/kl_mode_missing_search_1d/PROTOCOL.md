# Exploratory search for a forward/reverse mode-coverage separation

Requested 2026-09-23; amended: screening reverse L=1024; final reverse L=1048576.
Purpose: find an illustrative condition, not estimate average superiority across arbitrary targets.
All tested configurations and all paired seeds are retained, including failures and no-gap cases.

## Fixed learning algorithm
TRG semi-implicit actor: z~N(0,1), two 256-unit GELU layers, mu=10*tanh(head),
conditional Gaussian truncated to [-10,10], log sigma clipped [-5,-1], initialized -1.
Mean-head scale is variance_scaling initializer scale, not an output multiplier.
Adam 3e-4; N=M=128; 32 independent groups/update; temperature .25.
No target samples in training; actor proposals only. Forward is self-normalized
importance weighted marginal GMM NLL. Reverse is pathwise (score_L - grad log f),
with independent resampled density latents and stopped explicit density parameter gradient.
Same initialization and source/action RNG for paired methods, auxiliary bank independent.
Use the validated highest-precision density-bank MLP and max-scaled streaming score.
Forward has no L; reverse has finite-L ratio bias, so conclusions are about these estimators.

## Search grid, round 3 (config.json authoritative)
Rounds1/2: 24 cases,96 runs at10K yielded no reproducible strict separation.
Refine the transition between d=3.5,h=.5,scale1 (both fit) and d=5,h=.5,
scale1/10 (both lose outer modes). Do not discard the no-gap cases.
- d=3.75,h=.5, scale {.1,1}
- d=4,h=.5, scale {.1,1,10}
- d=4.25,h=.5, scale {.1,1}
- d=4.5,h=.5, scale {.1,1,10}
- d=3 and3.5,h=.4, scale1
Equal-mixture centers (-d,0,d), box [-10,10].
Q=.25*log(mean_k Normal(a;center_k,h^2)); normalize target over [-10,10].
12 cases x2 methods x2 seeds(0,1),10K updates. Log/evaluate0,1K,5K,10K.
Numerical source is unchanged from round2. Reuse its passed GPU gradient/score
validation after verifying identical source hashes; validate new target settings
on CPU. PRO6000 additionally passed the full GPU checks and may be used alongside
3090/4090 nodes; record device per run. No CPU fallback.

## Selection and confirmation
32768 actual actor actions; 512 equal bins on [-10,10], no KDE or smoothed density.
Basin boundaries are mode midpoints. Core is center +/- h. "Missing" requires
both basin mass <25% of target basin mass AND core mass <25% of target core mass.
Three separated peaks: every core >=50% of target core and each equal-width
valley interval has <=50% of the smaller adjacent core mass.
Prefer cases where both forward seeds have three peaks and TV<.15, while reverse
misses >=1 mode in both seeds; rank by reverse-minus-forward TV. If none, report
no qualifying condition and expand only in a separately committed follow-up.
Validate selected case on held-out initialization seeds 2,3 at 10K,L1024.
Final: both methods start FRESH from matched initial parameters, seeds0-3, 100K.
Reverse L=1048576 (chunk4096), forward unchanged. Do not resume small-L reverse
checkpoints into large-L and describe them as 100K of large-L training.
Report if large-L or extended training removes the selected difference.

## Execution and provenance
CPU mathematical checks; GPU parity/score/gradient/reference + timing preflight;
then independent Slurm case/seed jobs (1 GPU,2CPU), no unrelated run dependencies.
Full params/Adam/RNG checkpoints, signals save safely. Existing L1M study untouched.
Commit source/config/launch/protocol before execution. SHA+file hashes+jobIDs stored
beside immutable snapshot. Raw output and checkpoints sync to dildata, not Git.

## Focused follow-up: e02, 50K at screening L (2026-09-24)
At10K, e02 centers(-4,0,4),h=.5,mean init scale .1 passed selection on both
screen seeds0,1. Held-out seed3 also fits three peaks; seed2 covers the basins
but has poor shape (TV .295). All four reverse seeds miss both outer modes.
Extend ALL four paired seeds from10K to50K at unchanged reverse L1024 to
check whether the remaining forward shape error resolves and reverse missing
persists. This is an explicitly exploratory horizon extension, not a changed
criterion or exclusion of the unsuccessful seed. Seeds0,1 parents are under
runtime/screen; seeds2,3 under runtime/replicate. Preserve full Adam/RNG and
record parent source and checkpoint hashes. Same numerical update and parameters.
Final100K still uses reverse L1048576, with both methods fresh. Pair methods in
the SAME GPU allocation per seed (forward then reverse) to keep initialization
hardware identical. Do not initialize full confirmation from screening checkpoints.

## Persistence follow-up: e06 and e09
In e02, reverse seed2 recovered all modes by50K. Thus10K separation alone is
insufficient for a persistent-missing example. Retain this negative finding.
Test both adjacent candidates to50K, same L1024, seeds0-3:
- e06: centers(-4.25,0,4.25),width .5,mean head scale1.
- e09: centers(-4.5,0,4.5),width .5,mean head scale10.
Forward had all three basin masses at10K in both cases but still imperfect
local shape. Reverse had only the central mode in both screen seeds.
Seeds0,1 resume10K, seeds2,3 start fresh when no parent exists. Both methods
receive50K total updates; never omit poorly fitting seeds. Choose by all-seed
peak recovery and distribution error, not just coarse basin coverage.

## Round4: four-seed persistence screen
Prior follow-ups: e02 reverse recovers in3/4 by50K; e06 forward fails seed2,
reverse recovers seed3; e09 forward has poor central peak in seeds2/3, reverse
misses both outer modes in4/4. None is a clean four-seed example. Keep all results.
Screen eight conditions listed in config.json, ALL seeds0-3,50K updates each,
reverseL1024. This includes earlier e11 at a longer horizon plus intermediate
head scales and widths. Fresh initialization for both methods, sameGPU per pair.
Ranking: forward all4 seeds have three separated peaks and meanTV<.15; reverse
all4 seeds miss at least one mode. If no condition meets this, report partial
separation honestly and do not present the best seed as reproducible evidence.
Same numerical kernels and inherited GPU validations as previous rounds; runner
only generalizes screen seed/horizon indexing. Final100K reverseL1048576 remains
freshly initialized. No largeL confirmation is claimed until actually measured.
