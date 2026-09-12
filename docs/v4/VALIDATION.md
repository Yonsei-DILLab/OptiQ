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
