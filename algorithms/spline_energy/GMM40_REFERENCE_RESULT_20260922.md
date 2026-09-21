# Restored raw-energy GMM40 experiment, 2026-09-22

The restored state-free raw-energy implementation reproduced the preserved
successful GMM40 result. All five seeds were trained from initialization for
100,000 updates. This was not checkpoint reuse.

## Matched protocol

- DiKL GMM40 seed-0 target conditioned on `(-40,40)^2`
- rank 64, 129 knots, Adam `3e-4`
- 16,384 target queries per update
- proposal: 0.5 current policy + 0.5 uniform box
- 10,000 evaluation samples every 5,000 updates
- seeds 0, 1, 2, 3, 4
- fixed evaluation and sampling keys from the preserved experiment

The changed variable was the implementation: the run used
`StateFreeRawEnergyCircuit` and the shared restored raw-energy equations.

## Result

| Metric | Preserved mean ± SD | Restored mean ± SD | Mean difference |
|---|---:|---:|---:|
| Modes covered | 40.0 ± 0.0 | 40.0 ± 0.0 | 0 |
| Within target 3σ | 0.992280 ± 0.000327 | 0.992240 ± 0.000321 | -0.000040 |
| Forward KL | 0.0032561 ± 0.0010968 | 0.0032587 ± 0.0010972 | +0.0000027 |
| Reverse KL | 0.0032952 ± 0.0011516 | 0.0032970 ± 0.0011565 | +0.0000018 |
| Component-mass TV | 0.0259818 ± 0.0027210 | 0.0259631 ± 0.0027099 | -0.0000188 |
| MMD² | 0.00025955 ± 0.00001956 | 0.00025761 ± 0.00001646 | -0.00000193 |

Every seed first exceeded 90% 3σ mass at 10k and 95% at 15k. The 98%
crossings were `[25k, 20k, 20k, 20k, 25k]`, exactly matching the preserved
five-seed histories. The maximum seedwise forward-KL difference was `2.02e-5`.

## Provenance and limit

Training commit: `d543c93b6e18265b4524bbee8a04a5b382d5e8e7`.
SLURM array `2305944`; all five tasks completed with exit code 0. Runs used
RTX 3090 or RTX 4090 GPUs and took 1m43s–1m58s wall time including compilation,
evaluation, checkpointing and local W&B bookkeeping. W&B mode was offline and
all recorded run URLs are null.

The campaign, immutable snapshot, histories, checkpoints, samples, comparison
JSON and plot are under
`/scratch2/gsmin2024/research/optiq_spline_gmm_reference_20260922`.

This verifies the restored state-free two-dimensional fixed-oracle GMM40
implementation. It does not establish that the state-conditioned ordinary-TD
MuJoCo update learns successfully.
