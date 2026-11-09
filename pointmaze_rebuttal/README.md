# PointMaze rebuttal: isolated iBOLT / DIPO comparison

This folder is restricted to non-anonymous work, including `heechan-rebuttal`.
It must never enter an anonymous branch, snapshot or mirror. The runner rejects
anonymous/detached work branches and output paths inside anonymous worktrees.

```text
pointmaze_rebuttal/
  agents/ibolt.py        iBOLT adapter
  agents/dipo.py         native DIPO adapter, batched replay append only
  ibolt_core/           frozen learner + critic, independent of GMM40 changes
  envs/                frozen DrAC maps, physics, batch collector and metrics
  profiles/            independent method profiles and iBOLT learner defaults
  vendor/dipo/         unchanged BellmanTimeHut/DIPO source + MIT license + hashes
  evaluation.py        common rollout accounting, method-native action sampling
  run.py               one process / one method / one maze / one seed
  tests/               replay parity, real DIPO update, evaluation and boundary checks
```

The environment and iBOLT core are copied from OptiQ commit
`1c8d7a453d4de4c79ee60844ffdf8d0140f109a8`. The sole frozen-core adaptation is
the policy's relative import of its private critic. `SOURCE.json` records the
original file hashes. No mutable root `pointmaze/`, `ibolt/`, `models/`,
`configs/` or `gmm40/` modules are imported. The installed Python environment is
shared; existing GMM40 code, parameters, jobs and outputs are untouched.

DIPO is pinned to BellmanTimeHut/DIPO commit
`c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc`. Its six agent source files are
unchanged and checked against `vendor/dipo/SOURCE.json` before import.
Only replay append is vectorized. The native `DiffusionMemory.replace()` is
inherited literally: its advanced-indexed `np.copyto` does not persist improved
actions between minibatches. Action improvement within the current minibatch
still trains the diffusion policy. Do not silently substitute the historical
writeback adapter. Upstream model code has three 256-unit Mish hidden layers;
the adapter follows the code rather than the two-layer wording in its README.

## Profiles

| Setting | iBOLT | DIPO upstream U32 |
|---|---:|---:|
| Default temperature | 5 | not applicable |
| Parallel environments | 256 | 2048 |
| Transitions including warmup | 1,000,192 | 1,001,472 |
| Warmup transitions | 8192 | 8192 |
| Batch size | 4096 | 4096 |
| Updates per collection | 16 | 32 |
| Total learner updates | 62,000 | 15,520 |
| Replay capacity | 1M | 1M per native replay buffer |
| Actor / critic learning rates | 3e-4 / 3e-4 | 3e-4 / 3e-4 |
| Discount / target tau | .99 / .005 | .99 / .005 |
| Policy settings | N=M64, 256x2 GELU, log-sigma [-5,-1], initial -1 | diffusion100/cosine, action-gradient20 |
| Additional settings | beta1, mean init1e-4, DACER off | action LR .03, action clip .2, actor/critic clip2 |
| Evaluation action | fresh z each step, mu only | fresh initial Gaussian, reverse-process noise disabled |

These reproduce recorded method-specific collection profiles. They do not match
optimizer counts or compute budgets. DIPO's default is UTD1/64 and iBOLT's is
UTD1/16. An optimizer-budget-matched comparison must be named separately.
CLI overrides are recorded as actual values in config.json; changing budgets
is not the historical profile. Seed is explicit per run; no multi-seed campaign
or full training is launched by setup.

Both use original DrAC sparse +100 reward, 4D state, 2D bounded action, fixed
center resets and native termination. Simple/Medium/Hard have 4/4/8 goals and
150/300/600-step horizons. `TaskBatch` is synchronous in one process; policy
inference is batched, and physics is not GPU-accelerated or subprocess-based.
Executed clipped actions and pre-reset next observations enter replay. Both
success and timeout mask TD bootstrapping, preserving the recorded convention.

Evaluate normal and obstacle conditions every approximately 200K transitions
(rounded to a collection), 200 episodes each; final 500 each. Both methods use
the same evaluator and exact corrected removal SR5 / obstacle SR5 metrics.
Keep iBOLT `mu_only` separate from DIPO `policy`. All episodes, failures and
goal counts remain in raw NPZ files. Evaluation preserves training RNG state.

## Run from the repository root

Use the prepared Python 3.11 `.venv` with the Blackwell package profile.
No additional packages or old mujoco_py installation are needed for this adapter.

```bash
source scripts/activate-experiments.sh

# CPU adapter/boundary checks, plus real CPU DIPO optimizer update.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q pointmaze_rebuttal/tests

# Finite GPU execution checks: 8 transitions, 2 updates, five episodes per condition.
# These retain DIPO's full diffusion100 / action-gradient20 settings.
CUDA_VISIBLE_DEVICES=0 python -m pointmaze_rebuttal.run --method ibolt \
  --maze simple --seed 0 --smoke --output outputs/pointmaze-rebuttal/smoke-ibolt
CUDA_VISIBLE_DEVICES=1 python -m pointmaze_rebuttal.run --method dipo \
  --maze simple --seed 0 --smoke --output outputs/pointmaze-rebuttal/smoke-dipo

# Full runs: execute separately on available GPUs only.
CUDA_VISIBLE_DEVICES=0 python -m pointmaze_rebuttal.run --method ibolt \
  --maze hard --seed 0 --temperature 5 --output outputs/pointmaze-rebuttal/hard-ibolt-T5-s0
CUDA_VISIBLE_DEVICES=1 python -m pointmaze_rebuttal.run --method dipo \
  --maze hard --seed 0 --output outputs/pointmaze-rebuttal/hard-dipo-u32-s0
```

Before a full campaign, validate the actual vector/batch shape on an idle GPU:

```bash
# 256 environments, batch4096, warmup8192, 16 updates.
CUDA_VISIBLE_DEVICES=0 python -m pointmaze_rebuttal.run --method ibolt \
  --maze simple --seed 0 --steps 8448 --eval-every 8448 \
  --eval-episodes 5 --final-episodes 5 --output outputs/pointmaze-rebuttal/preflight-ibolt
# 2048 environments, batch4096, warmup8192, 32 updates.
CUDA_VISIBLE_DEVICES=1 python -m pointmaze_rebuttal.run --method dipo \
  --maze simple --seed 0 --steps 10240 --eval-every 10240 \
  --eval-episodes 5 --final-episodes 5 --output outputs/pointmaze-rebuttal/preflight-dipo
```

Default device is CUDA and missing GPU support is a hard error. Explicit CPU
validation can use `--device cpu`. Existing output directories are never
overwritten; use a new path for each run. Commands are foreground jobs, with
no implicit scheduler, automatic retry, GPU preemption or background campaign.
The caller chooses free GPUs and unique task/seed allocation. W&B is not enabled
by this runner; local files provide the experiment record. iBOLT CSV logs flush
every approximately 1000 updates and at evaluations. Native DIPO exposes no loss
return values; the adapter records update counts and evaluations without
fabricating loss metrics or modifying the upstream learner.

iBOLT defaults to deferred metric transfer in `agents/ibolt_logging.py`. The
adapter calls the unchanged frozen `IBOLT._step`, preserving replay sampling,
minibatches, actor/critic/target/Adam updates and RNG order. It retains only the
latest device metrics and transfers them together at the existing CSV flush
boundaries. The reference logger also overwrites these metrics until a flush;
the saved metric values are preserved. `--eager-metrics` selects the reference
per-update host reads for an A/B diagnostic, without changing the learner.
The optimized adapter also packs the five float32 minibatch arrays into one
host-to-device transfer and splits them on device using only slices/copies.
The frozen update receives the same input bits, shapes and dtypes; no arithmetic
or precision changes are introduced by this transport step.
Fixed gamma/tau/temperature/beta scalars are cached on device with their original
JAX dtypes. Each collection stages exactly the original sequence of replay
minibatches and executes the unchanged update sequentially with `jax.lax.scan`.
Actor, critic, target, Adam and RNG states carry from one update to the next;
batch size and UTD are unchanged. Only the last metrics survive, as in the
reference CSV logger. The runner records `update_execution` in each config.

Validate both modes on the same initialized models and replay data, excluding
compilation from the timed interval:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pointmaze_rebuttal.validate_logging \
  --updates 2048 --collection-execution scan \
  --output outputs/pointmaze-rebuttal/logging-parity-speed
```

This compares every serialized actor/critic/optimizer state leaf, RNG bits,
update counts, replay RNG state and CSV values for exact equality. Its timings
measure learner throughput; campaign progress supplies end-to-end throughput.

Each output contains actual settings, backend/package versions, a code-hash
manifest (including uncommitted source), progress, paired raw evaluations and
method-specific actor/critic/optimizer checkpoints. Final replay is saved unless
`--no-replay` is explicit. These snapshots support inspection, not exact resumed
training: simulator and all training RNG states are not serialized, and no
resume command is provided. A completed marker requires exact update counts,
finite model parameters and final evaluation. Smoke checks are not benchmark
results. Numerical-stack upgrades for RTX5090 can change historical results.

## Preparation validation, 2026-11-07

Both preparation servers passed all 9 dedicated tests and the CUDA smoke runs
for both methods (8 transitions, 2 real updates, normal + obstacle evaluation,
checkpoints and final replay). Additional default-shape preflights on server 1:

| Method | Environments | Batch | Transitions | Actual learner updates |
|---|---:|---:|---:|---:|
| iBOLT | 256 | 4096 | 8448 | 16 |
| DIPO | 2048 | 4096 | 10240 | 32 |

Both completed with GPU backends, finite parameters, five final episodes in
each evaluation condition and saved checkpoints/replay. Logs and immutable
per-run source fingerprints are under `outputs/pointmaze-rebuttal/validation/`.
These are execution/shape checks, not full 1M-transition benchmark results.
The 2048-env synchronous DIPO collector used approximately 37 GiB host RSS at
one observed construction point; this is not a peak-memory measurement.
