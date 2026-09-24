# Frozen-actor Monte Carlo score convergence for the selected 3-mode toy

User request 2026-09-24: show score on the y axis and log L on the x axis,
including the training bank L=2^20. This is evaluation only, no actor updates.

## Checkpoint selection (before examining MC results)
Use ALL eight final 100K checkpoints (forward/reverse x seeds0-3) of f05:
target centers(-4.25,0,4.25), width0.5, mean-head init scale3, N=M128,batch32.
Intermediate training checkpoints were overwritten, so no intermediate actor
is claimed. Seed0 is the prespecified main illustration; show all seeds too.
Parent numerical source commit5b2d79cd5aba408acaca469041e8cfb5cbfeb78f.
Read actor definitions from the original immutable snapshot. Record checkpoint
SHA256 before and after analysis. Never overwrite training files.

## Quantity being estimated
For fixed theta and action a, with IID z_l~Normal(0,1),
s_L(a)=sum_l k_theta(a|z_l)*(mu_l-a)/sigma_l^2 / sum_l k_theta(a|z_l).
This is d/da log q_L(a), not target score and not a parameter gradient.
k is the correctly normalized Gaussian truncated to[-10,10]. Its normalization
depends on mu/sigma, not the interior action, so the component action score above
is exact. Preserve float32 actor parameters/latents and highest dot precision as
in the training density bank; accumulate Gaussian density ratios in float64
with running max scaling. Check finite-bank score against autodiff and dense
evaluation; quantify float32-vs-float64 score for an identical bank.

## Banks and fixed actions
L=2^7 through2^24, all powers of two, 16 independent random streams. Within a
stream, banks are nested prefixes to expose how added samples change the ratio;
the16 streams are independent. MC latent RNG is independent of evaluation
actions and training. Save all raw scores, ESS and log densities, not just means.
Keep128 uniformly indexed actions from the existing32768-action final evaluation,
3 fixed empirical policy quantiles(10%,50%,90%), and9 prespecified target-region
locations. Actions stay unchanged for every L/repeat. Policy-domain and
low-policy-density external probes must be reported separately.

Four ADDITIONAL independent banks of2^24 give an empirical reference mean and
its standard error. This is not an exact integral, theorem, or ground truth.
Plot mean and10-90% repeat range, faint individual traces, independent reference
horizontal line/band, and vertical training L=2^20. Report RMSE to the independent
reference and sampling dispersion; do not divide by a near-zero pointwise score.
Show sensitivity at2^20,2^22,2^24 and all4seeds. Do not choose actions with good
convergence after seeing results or silently drop troublesome tails.

## Execution and storage
Commit source/config/protocol/launch before running, branchheejoon. Eight independent
GPU jobs,2CPU each, no training. Exact source/hash manifest and job IDs saved.
Raw results and checkpoint provenance to dildata, plots/Markdown locally and
dildata. Parent checkpoints remain only at existing experiment storage locations.
This only supports empirical finite-L stability on selected frozen checkpoints;
it does not prove training-time gradients or infinite-mixture scores are exact.
