# TRG 1D three modes: Forward KL vs Reverse KL

Status / completed runs / source commit / job IDs / snapshot time.

1. Task and settings: symmetric three-mode fixed Q, TRG truncated Gaussian, N=M128, batch32,20K updates,4seeds; forward1 condition and reverse4 L.
2. Figure: all final action histograms (32,768 actual samples), exact target, seed-wise panels.
3. Figure: histogram TV and mode mass vs update; mean±SD, individual seeds.
4. Figure: TV vs synchronized training seconds, separate JIT/diagnostic time.
5. Table: final histogramTV, basinTV, mode coverage, W1, backup error/SE, sigma, runtime.
6. Figure/table: same-checkpoint score RMSE vs L, repeated bank spread, 16K-to32K reference stability. Reference is not exact.
7. Interpretation: distinguish KL direction, finite density approximation, forward SNIS/proposal effects, and optimization. No claim all differences come from KL direction.
8. Limits: 1D fixedQ; no learned-critic/RL return inference, independent L bank still biased, N/M fixed does not equal compute.
9. Reproduction: source SHA, upstream hashes, resolved config, validation, checkpoint paths and raw sample/score files.

Incomplete runs remain explicitly incomplete; do not substitute early checkpoints as final20K results.
