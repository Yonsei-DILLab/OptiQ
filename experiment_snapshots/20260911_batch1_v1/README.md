# Batch-one OptiQ: GMM40 and frozen-Q, two requested versions

User request (2026-09-11): batch=1, temperature=1, proposal std=1; ver1 uses 16 actor samples and 4 random candidates plus 1 anchor per center; ver2 uses 2048 actor samples and 1 random candidate per center. Subsequent user correction: **frozen-Q retains the original truncated Gaussian KDE**; only GMM40 uses unbounded Gaussian KDE/actions.

## Frozen protocol

| Parameter | ver1 | ver2 |
|---|---:|---:|
| Batch | 1 | 1 |
| Actor samples | 16 | 2048 |
| Random candidates / center | 4 | 1 |
| Anchors / center | 1 | 0 |
| Total candidates | 80 | 2048 |
| OT matrix | 16×80 | 2048×2048 |
| Temperature | 1 | 1 |
| Proposal std | 1 | 1 |
| Density correction beta | 1 | 1 |
| Actor target | row-argmax | row-argmax |
| Actor loss | pointwise MSE | pointwise MSE |

- **5 training seeds, 0–4**, retained from the user's prior seed-budget decision. 13 existing frozen-Q cases plus GMM40 × 2 versions × 5 seeds = **140 runs**. Random initialization only; no coverage teacher or new 16D/32D cases.
- Each case/seed pair starts with identical actor parameters and optimizer state. Source hashes and initial state hashes are recorded and paired again in final analysis.
- **Frozen-Q:** original Q functions unchanged; truncated KDE with perturbation limit .5 and action bounds [-1,1]; 256×3 GELU; epsilon .05; 30 Sinkhorn iterations; IID Gaussian latent; Adam lr3e−4; 20k updates. Recompute the Boltzmann reference at **T=1** using independently converged float64 quadrature. Constant-Q remains valid on the bounded domain.
- **GMM40:** exact original 2D GMM40 log density, native coordinates, no clipping; 512×5 GELU; randomized 2D Gaussian latent grid for training, IID Gaussian for evaluation; 300 Sinkhorn iterations; epsilon .01→.0001 over the first 15k updates; lr3e−4 until 50k, then1e−4; total85k updates. Proposal std stays **1 throughout**. Actor sample count stays at the requested version's N throughout; no switch to4096. Gaussian mixture KDE and row-argmax/MSE come directly from audited `heechan-no-anchor` source.
- Equal updates within a case, **unequal compute/query budgets**. Log wall time, evaluated actions and random candidates. GMM and frozen-Q retain different remaining hyperparameters and target problems; this is not an isolated cross-domain ablation.
- The existing anchor weighting rule is retained: deterministic anchors are scored by the same KDE density correction. The campaign compares these implementations and does not assert an exact mixed-measure importance correction for anchors.
- No MuJoCo/other benchmarks, SAC runs, new architecture changes, or production checkout changes.

## Validation and outputs

Before training, independently reconstruct the actual sample cloud, weights, OT coupling, row-argmax targets and MSE on GPU for both versions on GMM40, 2D frozen-Q, and 8D frozen-Q. Verify candidate dimensions, anchor equality, bounded support, returned RNG and paired initialization. A CPU job recomputes all T=1 references; bounded Q expectation and partition integral must pass two consecutive grid refinements at tolerance1e−5. GMM reference uses1M exact independent target samples, with its Monte Carlo standard error reported.

Record early distributions after0/1/10/50/100 updates and subsequent checkpoints. Final evaluation uses100k IID actor samples (2,000 independent K=50 backup means); intermediate evaluations use20k. Compare mean Q and reference, backup bias/RMSE, mode-bin TV, effective modes, per-coordinate spread, and empirical sliced Wasserstein distance. GMM records mean log p, reference mean log p, absolute difference, mode coverage and occupancy. Reference and evaluation RNGs are separate from training RNG.

This campaign does not retrain SD2/SD3. Historical T=.25 baseline estimates are not reused as T=1 results.

Scheduler: at most2 concurrent GPUs. Separate campaign:

`login4:/lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260911_batch1_v1`

`setup_campaign.py` preserves the prior snapshot and overlays only the audited GMM-capable actor update and transport. `analysis_batch1/core.py` passes explicit configuration to that update, avoiding the prior cached-Hydra override failure. `reference.py` builds references; `smoke.py` validates; `worker.py` trains; `summarize.py` validates completed pairs and writes a standalone comparison report. `submit_campaign.py` requires both validation gates before submitting the140-task GPU array and its dependent final report job.

Launch IDs and live evidence are stored in `evidence/` once available. A submitted array is not evidence that all runs completed.
