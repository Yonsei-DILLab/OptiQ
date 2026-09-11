# Categorical OT target ablation — 2026-09-10

User request: `row-argmax말고 categorical로도 한번 돌려봐봐.`

- Preserve the production checkout and previous `20260909_v1` snapshot.
- Use a separate `20260910_categorical_v2` campaign snapshot. The first attempt (`v1`) was invalidated as described below.
- Change only `transport_target_mode` from `argmax` to `categorical`: one independent draw per normalized Sinkhorn row, then the original sample-specific MSE regression.
- Preserve original actor/proposal RNG streams using a folded-in target RNG. Reuse exact initial actor and optimizer checkpoints from the paired argmax runs; verify SHA256.
- Six representative frozen-Q cases: 2D eight modes, 8D separable, 2D four modes, 4D separable, unimodal, asymmetric 1D. Default and reference-informed coverage initialization; seeds 0–4; **60 runs**.
- Original τ=0.25, Sinkhorn ε=0.05 and 30 iterations, 16 actor samples × 4 candidates, 20k updates. No temperature/candidate-budget/iteration tuning.
- Additional distribution checkpoints after 1/10/50 updates; training RNG is independent of evaluation RNG. Original checkpoints and backup estimates retained.
- Primary outcomes: paired raw backup RMSE and bias (K=50), mode-bin discrepancy, action spread, effective modes, default versus coverage initialization. Secondary: K sweep and added TD noise.
- Validate argmax update parity, categorical sampling behavior, actual categorical update loss/finite parameters, and unchanged returned RNG before launch.
- At most two concurrent GPUs; no seed expansion, SAC comparisons, MoveCar retraining, or other benchmarks in this ablation.
- On completion the dependent CPU job validates all 60 runs and generates paired CSV/JSON, figures, and a standalone HTML report. Interpretation will distinguish target sampling from actual actor-distribution recovery.

`setup_campaign.py` creates the isolated code snapshot; `categorical_smoke.py` validates it; `categorical_worker.py` trains one task; `categorical_summary.py` builds a paired full or partial summary.

## Current valid v2 launch

- Campaign: `login4:/lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260910_categorical_v2`
- GPU validation **2194389 passed** on all three representative problems. The exact frozen-Q wrapper produced the same loss and updated parameters as direct categorical and different parameters from argmax. Independent loss reconstruction matched on GPU.
- Dispatcher **2194397** submitted GPU array **2194399** (60 tasks, concurrency 2) and dependent summary **2194400**.
- Actual categorical mode and the first full 20k-update run (2D eight modes, default, seed 0) verified on node23. Subsequent tasks are running on node10/node23. All 60 runs have not yet completed.
- `evidence_v2/` contains protocol, task list, jobs, and GPU validation evidence for this valid run. Final comparisons will be generated remotely under `comparison/`.

## Invalidated v1 launch record

- Remote campaign: `login4:/lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260910_categorical_v1`
- CPU validation **2194143 passed**. Original argmax parameters/loss/RNG unchanged; categorical loss independently reproduced. Frequencies for [0.1,0.2,0.7]: [0.10075,0.19872,0.70053].
- GPU array **2194160**, 60 tasks at concurrency 2. Actual GPU execution verified on node03 and node15.
- Dependent CPU validation/summary **2194161**. Outputs will be written under the campaign's `comparison/` directory, including standalone `report.html`.
- `evidence/` preserves the actual protocol, task list, scheduler commands, and smoke results. These are launch records, not evidence that all training runs have completed.

**Do not use v1 as categorical evidence.** `lru_cache` treats `config()` and `config(0)` as separate entries. The worker initially set the seed-specific configs, but the real frozen-Q wrapper read the unchanged no-argument config and still dispatched argmax. Identical paired results exposed the issue. Array 2194160 and gate 2194161 were canceled; outputs were preserved and flagged with `INVALID_VARIANT.json`. The production checkout and original argmax campaign were unaffected.

The corrected `configure_categorical()` sets both call forms explicitly. The smoke test now calls **the exact shared.ot_update wrapper used by frozen-Q training** and checks that loss and updated parameters match the direct categorical call and differ from argmax. V2 starts from the original valid argmax campaign, not from v1 outputs.

### Checkpoint validation correction

The coverage warm-start checkpoint stores Dense dictionary entries as `bias, kernel` after JIT, while restoring it into a fresh actor template serializes them as `kernel, bias`. A raw file hash therefore produced a false mismatch after otherwise completed training. A recursive comparison found **zero tensor/value/dtype/shape differences**. Validation now checks the original reference file hash and exact canonical checkpoint hashes after sorting dictionary keys only. Training code and parameters were not changed, and completed training results are reused. The summary revalidates and marks previously completed runs affected by this metadata check.
