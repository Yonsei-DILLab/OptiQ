# v3 implementation validation

Validated on this instance with `/root/.venv-optiq-mujoco/bin/python`, JAX CPU,
Gymnasium/MuJoCo, and single-thread BLAS. This is implementation validation,
not a v3 benchmark or a GPU performance measurement.

## Results

- **13 v3 tests passed** in `tests/test_v3.py`.
- **84 existing regression checks passed**: 79 tests covering v2 final settings,
  reference arithmetic, semi-implicit/scalar critics, OT NLL and soft guards;
  plus five v2 instance-path tests.
- Real Humanoid-v4 and Ant-v4 integration checks each ran eight environment
  steps with a two-step warmup and **six critic + six actor updates**.
  These retain the actual 256×3 networks and 16×64 OT but use replay batch 4.
- Both integration checks replaced the policy entropy and soft-guard functions
  with functions that raise if invoked, and cleared JIT caches before tracing.
  Training completed without calling them. Proposal density remains active.
- Actor and critic TrainStates, including optimizer state, were restored from
  checkpoints and compared. Restored actor samples matched under the same RNG.
- Time-limit truncations continued to bootstrap; true-terminal TD masking was
  checked independently with known target values.
- The launcher passed `scripts/run_v3.sh 0 --check benchmark=ant`; shell syntax
  and `git diff --check` passed.

The initial integration assertions read an empty CSV flush as the last training
row. The assertion was corrected to inspect actual training rows, then all v3
and path tests passed (18/18). No learning-code change was needed for that fix.

## Mathematical and control-flow checks

1. An action-dependent twin critic checks `r + gamma*(1-done)*min(Q_target)`
   against an independently assembled expected target and MSE.
2. Changing the teacher-temperature argument from .1 to 17 and the unused
   entropy sample count from 0 to 91 leaves the plain-TD critic update identical
   for a fixed actor/batch/RNG. Target gradients into actor parameters are zero.
3. Entropy coefficient and backup entropy terms are zero; policy entropy
   estimates and soft-guard gap metrics are absent from v3 training logs.
4. Disabling actor entropy diagnostics preserves the full OT NLL loss and the
   resulting actor/optimizer update to numerical tolerance.
5. v3 resolved algorithm settings differ from v2 only in explicit TD mode,
   zero entropy coefficient, disabled entropy evaluation and disabled guard.
6. Conflicting entropy/guard settings are rejected. The v3 launcher validates
   the actual Hydra overrides and rejects an entropy diagnostic or MSE variant.
7. Existing v2 configs, soft TD arithmetic, conditional proposal, NLL gradients,
   guard behavior, scalar TD and baseline checkpoint tests remain covered.

## Commands used

Environment for pytest:

```bash
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Initial combined verification (79 baseline tests passed; two v3 CSV assertions
were subsequently corrected as described above):

```bash
/root/.venv-optiq-mujoco/bin/python -m pytest -q \
  tests/test_v3.py tests/test_v2_final.py tests/test_semi_implicit.py \
  tests/test_scalar_critic.py tests/test_distributional_distillation.py \
  tests/test_soft_guard.py tests/test_soft_improvement.py
```

Final v3/path verification, **18 passed**:

```bash
/root/.venv-optiq-mujoco/bin/python -m pytest -q \
  tests/test_v3.py tests/test_v2_instance_paths.py
bash -n scripts/run_v3.sh scripts/supervisor_v3.sh
scripts/run_v3.sh 0 --check benchmark=ant
git diff --check
```

No online W&B run, installed supervisor service, or 1M-step v3 experiment was
created by these checks. Full training performance remains to be measured.
