# HalfCheetah GMM-reference campaign, 2026-09-22

## Hypothesis and controlled comparison

The GMM40-restored raw-energy circuit can learn HalfCheetah with ordinary TD
while retaining one-forward exact spline sampling.

Baseline: the completed legacy Spline Energy runs at commit `37dbbea`, seeds
0–2, 1M environment steps. Final native stochastic-policy returns were
5842.85, 5964.58 and 5531.45.

Changed method bundle: `--profile gmm_reference`, which selects the GMM40 raw
energy parameterization, rank 64 / 129 knots, ordinary TD, MSE and no gradient
clip. This is the combined restored reference profile documented in
`algorithms/spline_energy/DIRECT_ALIGNMENT.md`; it is not a one-variable
ablation. Environment, replay, warmup, batch size, Adam learning rate, target
EMA, temperature, evaluation episodes/seeds and update-to-data ratio remain
fixed.

Primary metric: `eval/mean_reward` every 10k steps and its 0–1M AUC. Secondary
diagnostics: finite training, wall/training speed, sampled absolute TD error,
Q/target scale and pre-clipping gradient norm. Three seeds are required for the
performance comparison.

## Execution gate

Run a seed-0 10k GPU smoke first. It must create `COMPLETE.json`, reach 10k
without a nonfinite metric or GPU memory failure, and record offline W&B only.
Its reward is diagnostic and is not used as a performance selection gate.
Then launch seeds 0–2 for independent 1M runs from fresh initialization.

All source is snapshotted from a clean committed tree. W&B stays offline and no
automatic cloud sync is performed.
