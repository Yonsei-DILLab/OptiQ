# Annealing validation, 2026-09-12

CPU validation in the existing pinned MuJoCo environment:

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/root/.venv-optiq-mujoco/bin/python -m pytest -q \
  tests/test_temperature_schedule.py tests/test_v4.py \
  tests/test_behavior_uniform.py tests/test_v3.py --disable-warnings
```

Result: **54 passed, 13 warnings in 49.26 seconds**. These are validation
tests, not completed Ant performance experiments.

- All four requested configurations differ from v4 only in initial teacher
  temperature, schedule, uniform probability, and recording metadata.
- The 20K/40K schedules retain T=10 through warmup, use geometric midpoint
  sqrt(2.5), reach exactly .25 at 25K/45K, and hold through 1M.
- Reject invalid schedule durations/endpoints and enabled soft-TD schedules.
- Actual short Hopper training, for both p=0 and p=.1, executes six updates
  and checks the temperature returned by the real JIT teacher update against
  an independent geometric sequence, including three final-T updates.
  Both evaluation files and paired seeds are verified; config is unchanged.
- Existing v4 collection/replay and isolated dual-evaluation tests pass.
- Existing v3 tests verify temperature-independent plain TD targets, stopped
  target gradients, and absence of entropy/guard computations.

Ongoing experiments use separate frozen source directories and were not
modified or restarted by this validation.
