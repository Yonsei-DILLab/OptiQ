# Final selected-toy histogram TV using 2^20 actual actions per seed

User request2026-09-24: replace32768-action TV evaluation with1048576 actions,
and show a publication-style two-panel figure: leftforward, rightreverse,
averaged over all4seeds. Evaluation only; no training or checkpoint changes.

Use all8 final100K f05 actors from parent source5b2d79cd5aba408acaca469041e8cfb5cbfeb78f.
Retain original512 equal-width bins over[-10,10], exact target bin probabilities
from its Gaussian CDF, same actual truncated-Gaussian actor sampler. New independent
evaluation RNG,32 chunks x32768, each action has a freshly sampled latent and
conditional Gaussian noise. All parameter and sample computations preserve the
original float32 inference implementation. Reusing an N-component finite bank
is NOT allowed; evaluation samples the full implicit actor independently.

TV_seed=0.5*sum_bins(abs(actor bin frequency - exact target bin probability)).
Report mean and sample SD of the4per-seedTV values. The figure is the pointwise
mean of4normalized sample histograms with a +/-1SD seed band; no KDE or smoothing.
Also report TV of the averaged histogram, but do NOT substitute it for mean
per-seedTV: averaging different actors can artificially improve distribution fit.
Keep32768-sample metrics as historical measurements, not overwrite raw runs.
Include matched subset(32K prefix)TV for sampling-size context, without retraining.
Figures share x/y scales and full support, exact target dashed, learned density
as a step histogram. Export vectorPDF/SVG and600dpiPNG; all4seeds equally weighted.

Before execution commit exact code/config/protocol/launch onheejoon. Record full
source/checkpoint hashes, jobIDs. Use a known-good login4 computeGPU, no login-node
training. Parent checkpoints untouched; samples+metadata on dildata, no generated
samples/checkpoints inGit. Persist per-seed histograms and raw actions separately.
