# v4 validation — 2026-09-12

In an isolated checkout, using the existing pinned v3 environment on vast1
with `JAX_PLATFORMS=cpu`:

```text
python -m pytest -q tests/test_v4.py tests/test_v3.py --disable-warnings
16 passed, 13 warnings in 31.34s
```

The new tests check both conditional-mean action definitions, independence
from the conditional std head, preservation of stochastic collection, paired
episode reset seeds, both saved evaluation files and logging namespaces, and
restoration of collection RNG and evaluation flags even after an exception.
The integration check uses short real Hopper episodes and eight training steps.
It verifies six critic/actor updates; it is not a performance benchmark.

`bash scripts/run_v4.sh 0 --check benchmark=hopper` also passed and resolved
256x2 actor/critic, T=.1, initial sigma=.5, 1M steps, and dual mu-only evaluation.
`bash -n scripts/run_v4.sh` and `git diff --check` passed. Existing GPU training
runs were not modified by this branch implementation.

## Optional 10% uniform collection profile — 2026-09-12

```text
python -m pytest -q tests/test_v4.py tests/test_behavior_uniform.py tests/test_v3.py --disable-warnings
38 passed, 13 warnings in 39.44s
```

All eight Ant configurations (T=.1/.25, seeds 0..3) differ from v4 only in
the collection replacement probability and recording fields. The profile
and alias resolve identically. The existing uniform-sampling tests check
the empirical 10% rate, whole-vector replacement, RNG isolation, warmup and
non-unit action bounds. Real Hopper integration tests exercise v4 learning,
executed-action/replay agreement and both evaluation modes with the collector
disabled or forced to replace every post-warmup action. The latter is a short
test setting; the experiment profile uses .1. Evaluation bypasses the
collection hook and preserves its RNG, including on an evaluation exception.

The actual `run_v4.sh --check` launcher also passed for
`OPTIQ_CONFIG=mujoco_v4_behavior010`, Ant, T=.1: actor/critic 256x2,
uniform collection p=.1, dual mu-only evaluation, project `v4-test`.
The running baseline's source hashes and all eight queued configurations
were verified unchanged. No new performance run was launched by these checks.
