# Collection exploration validation — 2026-09-11

CPU checks, with the existing pinned MuJoCo Python and an isolated directory
containing sklearn 1.7.2, joblib 1.5.2 and threadpoolctl 3.6.0:

```text
pytest -q tests/test_collection_exploration.py tests/test_exploration_queue.py tests/test_v3.py
40 passed (33.93 seconds)
```

Coverage includes config parity with v3; entropy estimation conditional on
state; correct scalar-alpha feedback; executed/replay action consistency with
non-unit action bounds; warmup and evaluation isolation; unchanged actor and
learner/rollout RNG states during entropy estimation; real updates and saved
exploration state; rejection of partial baseline evidence; equal four-seed
temperature selection; exact-tie rule; all-16 common-temperature configs;
and a simulated controller waiting for the final baseline run then executing
all 16 jobs with at most four workers and no overlap on a GPU.

A separate **validation-only CPU Ant-v4** run exercised the actual entrypoint
factory with original 256x3 actor/critic networks: 12 environment steps, 10
critic and actor updates. A production-shape entropy probe used 256 states x
200 actions x 8 dimensions (states repeated from the short validation buffer).
The GMM fits all converged; every diagnostic was finite. The entropy probe
took 3.42 seconds on two CPU cores. Positive entropy error decreased noise
standard deviation from .15 to .14557, as expected. Actor, critic and
exploration checkpoints were written successfully.

These are implementation checks, not completed performance experiments.
No GPU training worker was interrupted or used for these checks. Baseline
training code, configuration, queue and shared Python environment were not
modified. GPU production execution remains subject to the 20-run completion
dependency; the controller checks online W&B identity and metrics after each
future launch.

Full validation records are outside the repo:
`/root/anal/v3_exploration_auto_20260911T193612Z/`.
