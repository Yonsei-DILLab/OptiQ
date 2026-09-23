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

## Search grid (config.json authoritative)
Equal-mixture centers (-d,0,d), Gaussian target width h:
- d=1.5,h=.25, mean init scale {1e-4,.01,.1,1}
- d=2.5,h=.5, mean init scale {1e-4,.01,.1,1}
- d=3.5,h=.5, mean init scale {.01,.1,1}
- d=5,h=1, mean init scale 1 (previous condition reference)
Q=.25*log(mean_k Normal(a;center_k,h^2)); normalize target over [-10,10].
Only centers, h and mean initializer change. Actor sigma bounds and proposal do not.
12 cases x 2 methods x 2 seeds(0,1), 10K updates each. Log/evaluate 0,1K,5K,10K.

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
