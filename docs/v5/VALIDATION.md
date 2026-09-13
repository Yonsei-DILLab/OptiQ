# v5 implementation validation

2026-09-13. Local branch `v5`, based on v4 commit `558ad09`.
Validation used CPU only; no production worker, queue, frozen source or
checkpoint was modified. No new benchmark or W&B training run was started.

## Regression and integration tests

Environment: `/root/.venv-optiq-mujoco/bin/python`, JAX 0.4.33, Flax 0.9.0,
Gymnasium 0.29.1, MuJoCo 2.3.7. Run from the v5 worktree:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/root/.venv-optiq-mujoco/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_v5.py tests/test_v4.py tests/test_v3.py \
  tests/test_conditional_proposal.py tests/test_distributional_distillation.py \
  tests/test_temperature_schedule.py
```

**58 passed, 13 warnings in 75.13 seconds.** Warnings were existing
Matplotlib/Pyparsing deprecations.

Checks include:

- v5 default and alias match the completed T=.25 ablation's numerical recipe;
  v4 retains the default `sample` mode.
- Invalid OT mode and conflicting v5 launcher settings are rejected.
- Independently reconstructed mean-action cost, Gaussian teacher density,
  beta=1 weights, Sinkhorn rows and full-row NLL match the implementation.
- Teacher density, ESS and next RNG agree between OT modes on identical inputs;
  both actor heads update and critic gradients remain stopped.
- A real short Ant-v4 loop verifies `train → _train → update_actor` receives
  `mean`, performs six updates after two warmup steps, uses plain TD, and saves
  both paired evaluation modes. Time-limit transitions still bootstrap.
- Existing v3/v4 integration, conditional proposal, Gaussian NLL and optional
  temperature schedule regressions pass.

The short validation uses reduced episode lengths and training budgets; it
does not measure benchmark performance.

## Exact parity with the completed experiment

Each of four saved 50K Ant actor/critic states was evaluated on 16 saved
observations. For identical parameters, optimizer state, observations and RNG:

| v5 path | Frozen reference | Cases | Maximum absolute difference |
|---|---|---:|---:|
| `ot_student_action=sample` | Original v4 source `6f5c987` | 4 seeds | **0.0** |
| `ot_student_action=mean` | Completed meanOT experiment source | 4 seeds | **0.0** |

Actor parameters, optimizer state, NLL, next RNG and every returned metric were
**bitwise equal** in all eight comparisons. This checks one update from each
saved state, not bitwise equality of an entire future training run.

Machine-readable [parity record](validation/parity.json) includes input/source
hashes. The [validation script](validation/parity.py) records the local frozen
source paths and checkpoint protocol; reproducing that audit requires those
preserved experiment files. Portable semantic tests live in `tests/test_v5.py`.

## Entry points and documentation

`scripts/run_v5.sh 3 --check benchmark=ant` validates the real launcher and
resolves the T=.25, mean-OT, no-uniform/no-annealing profile without training.
The direct `run_optiq_dime.py --cfg job --resolve benchmark=ant` entry also
defaults to `mujoco_v5`. Explicit v4 configuration files remain unchanged.

[Checks and file hashes](validation/verification.json) distinguish this code
validation from the [completed precursor experiment](../v4/MEAN_OT_RESULTS_KO.md).
