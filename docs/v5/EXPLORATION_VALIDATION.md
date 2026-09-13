# v5 exploration profile validation

Validated on 2026-09-13 using `/root/.venv-optiq-mujoco/bin/python`, CPU JAX,
MuJoCo Ant-v4, and no W&B training runs for the short checks.

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/root/.venv-optiq-mujoco/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_v5_exploration.py tests/test_v5.py \
  tests/test_temperature_schedule.py tests/test_behavior_uniform.py
```

Result: **55 passed in 53.85s**. The 13 warnings are existing matplotlib
pyparsing deprecations. The checks cover all three profiles and four seeds,
profile aliases, the canonical v5 restrictions, existing annealing/uniform
behavior, and three short Ant integration runs. Those integration runs check
mean-action OT forwarding, zero entropy in TD backup, actual temperatures,
executed/replayed action agreement, both epsilon=0 evaluation modes, and
isolation of evaluation RNG from collection.

All three `scripts/run_v5.sh 0 --check benchmark=ant` invocations also passed
with their respective `OPTIQ_CONFIG` values. These resolve configuration and
exercise the actual launcher without starting training.

The following files are byte-for-byte unchanged from initial v5 commit
`38f48bd`: `optiq_dime/algorithm.py`, `optiq_dime/policy.py`,
`optiq_dime/temperature.py`, and `run_optiq_dime.py`. The new profiles reuse
the existing collection replacement and temperature schedule implementations.
Changes here are profiles, profile validation/selection, tests, and documents.

These checks establish configuration and execution behavior. They do not
establish an improvement in learning performance; the requested three
four-seed, 1M-step cohorts provide that comparison. Queue manifests, frozen
source hashes, controller handoff validation, and worker status are retained
under the instance's `/root/optiq-experiments` campaign directory.
