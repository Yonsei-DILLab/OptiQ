# Measured GMM40 result

The final comparison trained five seeds for 100,000 updates with rank 64,
129 knots, Adam 3e-4, a 0.5 policy / 0.5 uniform defensive proposal, and
16,384 target-value queries per update. Evaluation used 10,000 samples from the
bounded DiKL GMM40 seed-0 target on `(-40, 40)^2`.

| Metric | Five-seed result |
|---|---:|
| Modes covered | 40/40 for every seed |
| Within target 3-sigma regions | 99.228% +/- 0.033% |
| Forward KL | 0.003256 +/- 0.001097 |
| Reverse KL | 0.003295 +/- 0.001152 |
| Component-mass TV | 0.02598 +/- 0.00272 |
| MMD^2 | 0.0002595 +/- 0.0000196 |

All seeds first exceeded 90% high-density mass at 10k updates and 95% at 15k.
Three seeds exceeded 98% at 20k and the other two at 25k. The reported values
come from the preserved campaign summary at
`optiq_latest_direct_comparison_20260921/analysis/mu90_100k_a1/summary.json`.

Training commits recorded by the five run manifests were
`19927e310866fc7f6b8fc7ba5a374fc384746b93` for seed 0 and
`8f6100de4a8f2ae636654c5b3ad5932b0494a79e` for seeds 1-4. The latter is the
frozen source copied here; the campaign verified the GMM40 implementation and
dependencies against the private upstream commit
`811e0e2e59a0c9137f38433e4fc18adaf5e6a8fa`.

W&B project: `OptiQ/OptiQ-GMM40-Sampling-Comparison`.
