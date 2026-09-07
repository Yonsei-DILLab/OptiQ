# OptiQ parameter sweep

Branch: `parameter-sweep`, based on the DIME-integrated `skewed-proposal`
commit `dbc2686`. This is a preparation branch, not a deployment or launch.
Existing server jobs retain their original code, configs and 5K evaluation cadence.

## Configuration ownership

- `configs/alg/optiq_dime.yaml`: original actor/critic/optimizer defaults.
- `configs/parameter_sweep.yaml`: shared protocol, explicitly symmetric stratified
  sampling (the parent actor config defaults to skewed sampling).
- `configs/sweeps/parameter_sweep.json`: seeds, factorial axes, benchmark IDs and
  explicit task-specific value supports.
- `scripts/parameter_sweep.py`: dependency-free dry-run command/JSONL generator.

## Common protocol

| Setting | Value |
| --- | --- |
| Budget / seeds | 1M environment steps / 0, 1, 2 |
| Warm-up / replay / batch | 5K / 1M / 256 |
| UTD / actor update | 2 critic updates per env step; actor every critic update |
| Actor | One-step implicit MLP, 256 × 3 |
| Critic | DIME twin categorical critic, 2048 × 2, 101 atoms, Batch Renorm |
| Discount / critic entropy coefficient | 0.99 / 0.005 |
| Optimizer | Adam, actor and critic LR 3e-4; actor β1=0.9, critic β1=0.5, both β2=0.999 |
| Target update coefficients | Critic `alg.tau=1`, actor `alg.policy_tau=1` (NOT weighting temperature) |
| Reference sampling | Symmetric stratified truncated Gaussian; normalized action box |
| Centers / random candidates | 16 / 64 |
| Source Q / OT | Twin mean; normalized squared cost; entropic Sinkhorn ε=0.05, 30 iterations; row argmax targets |
| Density weighting | `Q / temperature - beta * log(q_KDE)`; correction enabled; fixed beta, not adaptive |
| TD smoothing | std 0.2, clip 0.5, unchanged when proposal sigma changes |
| Evaluation | Every **10K**, 10 stochastic episodes; evaluate at start |
| Training diagnostics | First training update, then at 1K-step bucket crossings; not a 1K-step average |
| Checkpoints | First training update and every 50K; actor/critic states, NOT a full replay/environment resume |
| W&B | `online-optiflow/optiq_parameter_sweep`; unique run names, task groups |

The resolved Hydra configuration is uploaded with each W&B run. No API keys
belong in configuration or source control.

## Sweep axes

| Axis | Values |
| --- | --- |
| Proposal sigma | 0.1, 0.2 |
| Weighting temperature | 0.1, 0.25, 1.0 |
| Density-correction beta | 0, 0.001, 0.1 |
| Anchors | Off: N16 × R4 = 64 random; on: N16 × R5 = 64 random + 16 anchors |

Clip is always **2.5 × sigma**, not another axis. DIME critic is fixed.
This is 36 configurations × 3 seeds × 4 tasks = **432 planned runs**.
The anchor comparison also changes the total candidate budget (64 vs 80),
so it is not a strictly compute-matched ablation.

Beta is a density-correction/regularization mixing coefficient, **not itself
a KL constraint radius**. With a genuine reference sample the corresponding
population target is proportional to `exp(Q/T) q_KDE^(1-beta)` on proposal support.
Explicit deterministic anchors are additional candidate heuristics, not IID
samples from a continuous KDE. This branch does not change that existing
estimator or claim exact population-target recovery with anchors.

## Benchmarks and exceptions

| Task | Environment ID | Categorical support | Status |
| --- | --- | --- | --- |
| MuJoCo Humanoid | `Humanoid-v4` | [-1600,1600] | Reset/step smoke passed; full training pending |
| DMC Dog run | `dm_control/dog-run` | [-200,200] | Existing benchmark |
| DMC Humanoid run | `dm_control/humanoid-run` | [-200,200] | Existing benchmark |
| MyoSuite pen-twirl-hard | `myoHandPenTwirlRandom-v0` | [-200,200] **provisional** | Dependencies, environment and reward/value support require validation |

Gym support follows this repository's DIME gym recommendation. MyoSuite is
optional and must be validated in an isolated compatible environment: do not
upgrade MuJoCo/JAX in a running experiment environment. The hard task mapping
is documented in the [official MyoSuite suite](https://myosuite.readthedocs.io/en/stable/suite.html).
Do not interpret a generated MyoSuite command as a validated runnable benchmark.

## Inspect without launching

From the repository root:

```bash
python scripts/parameter_sweep.py
python scripts/parameter_sweep.py --tasks dog-run --seeds 0 --format commands
python scripts/parameter_sweep.py --format jsonl
```

For configuration-only inspection, append `--cfg job --resolve` to a generated
training command (requires the installed training dependencies). Jobs should
eventually be dispatched one per GPU by the existing supervisor/queue mechanism;
this generator does not start processes, allocate GPUs, retry jobs, or upload data.

## Optimization and measurement

`diagnostics_interval=1000` removes detailed ESS/Q/geometry statistics and six
counterfactual-temperature calculations from ordinary updates using a static JAX
branch. Logged scalar metrics are packed into one device vector before the host
copy. All proposal sampling, density weights, critic/actor losses, Sinkhorn work,
optimizer updates, UTD and RNG progression remain unchanged.

`diagnostics_interval=1` restores every-training-step diagnostics; `0` disables
training diagnostic scalars. Old configs without the field retain cadence 1.
The sparse setting compiles two update variants (diagnostics on/off), so initial
compile time can increase. SB3 retains its existing logger dump cadence: the
diagnostic interval is computation cadence, not a guarantee of exact W&B points.

Moving evaluation from 5K to 10K halves scheduled evaluation calls (200 to 100
over 1M steps; initial evaluation unchanged). It does **not** halve training time.
Large candidate-critic evaluation still dominates the arithmetic workload; no
measured GPU speedup is claimed without a matched post-compilation benchmark.

Regression tests compare actor, critic, optimizer, Batch Renorm state and RNG
with diagnostics on/off using CPU-sized networks, including fixed/adaptive beta.
These tests are not a full-sized GPU performance measurement.

Validation on 2026-09-07 used JAX 0.4.33, `JAX_PLATFORMS=cpu`, hidden GPU
devices and two pinned CPU cores in an isolated temporary copy on vast1.
All **20 tests passed** (`python -m pytest tests -q`, 73.69 seconds), including
the existing proposal-density normalization tests. Only headless GLFW and
deprecated JAX clip-argument warnings were emitted.
Gym Humanoid, DMC Dog run and DMC Humanoid run all passed reset/step smoke tests;
all four benchmark command configurations composed successfully. A 12-step
Pendulum smoke test exercised the host training loop and verified 16 optimizer
updates after four warm-up steps (UTD=2).
