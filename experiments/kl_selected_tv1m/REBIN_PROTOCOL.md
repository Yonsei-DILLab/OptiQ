# 4096-bin reevaluation and standard Matplotlib figure

User request2026-09-24: simplify plotting style and investigate whether the
512-bin evaluation explains TV around0.04; evaluate with4096bins as well.

No training, new checkpoints or new actor actions. Use the identical saved
1048576 actual float32 actions for every seed/method from evaluation source
8b35c3ecae257846ee453e099af2432064e894f0. Hash the input action files.

Evaluate512,1024,2048,4096 nested equal-width histograms over[-10,10]. Compute
exact target probabilities with the Gaussian CDF, no midpoint quadrature.
For each fixed sample, nested histogram TV must be nondecreasing (up to numerical
roundoff). Verify512-bin TV reproduces prior values. Preserve mean per-seedTV,
sampleSD, and separately TV of the averaged histogram. Mode-basinTV is computed
from the same sample, using midpoints(-2.125,2.125), to distinguish coarse mode
mass matching from within-mode distribution fit. Plot with standard Matplotlib
default colors/fonts/spines, a full-size12x4.6inch two-panel layout, step histograms,
exact target dashed, mean+/-seedSD, common axes. No KDE or hand-smoothed actor curve.

To calibrate finite-sample effects, for each bin count draw128 multinomial
histograms with sample_count=1048576 and probabilities exactly equal to the target
bin masses (PRNGseed202609242). These are the EXACT count distribution of a
perfect target sampler. Measure histogramTV to its own target probabilities.
Report mean and5-95% repeat quantiles as a sampling-noise calibration, NOT a
universal lower bound and NOT a bias correction that can be subtracted from
actorTV. The reference has no actor error. This does not change training.

User addition: also measure Wasserstein distance. Use1D Wasserstein-1 in original
action units (no binning): sort each1M action sample and compare to the normalized
target inverseCDF at midpoint ranks(i+0.5)/n. Solve target quantiles by42 bisection
steps on[-10,10], reuse the SAME exact-target quantiles for allactors. Midpoint
quantile integration differs from the empirical-to-target W1 integral by at most
support_width/n=20/1048576<2e-5, apart from inverseCDF floating-point error.
Do not compare to a separately sampled target as the main W1 metric. Calibrate
finite-sample W1 with16 independent perfect-target action draws(PRNG202609243),
using correctly weighted truncated-Gaussian mixture sampling. Report W1 perseed,
mean/SD across4seeds, perfect-sampler mean and5-95%range, and annotate mainfigure.

Commit this protocol and executable before evaluation. Source checkpoint and
raw-action sampling provenance remain unchanged. Archive512-bin custom-style
figure before replacing report's representative figure. Store all new counts,
JSONmetrics, figuresPNG/PDF/SVG andMarkdown on dildata, outsideGit.
