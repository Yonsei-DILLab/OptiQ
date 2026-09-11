# Final v2: implementation and experiment reproduction

The final entry is `mujoco_v2` (equivalently `mujoco_v2_checked` or `v2/final`).
Its definition is [configs/v2/final.yaml](../../configs/v2/final.yaml).
The completed training algorithm is pinned to `8cb4f23`; this release reorganizes
configuration/document paths and preserves the common learning implementation.
See [MANIFEST.json](MANIFEST.json) for source and selected-config provenance.

## 1. On another Linux GPU machine

Use a checkout containing these files and Python 3.11. The completed runs used
an RTX 3090, JAX/JAXlib/CUDA plugin 0.4.33, Flax 0.9.0, Optax 0.1.7,
NumPy 1.26.4, Gymnasium 0.29.1, MuJoCo 2.3.7, SB3 2.1.0 and W&B 0.29.0.
The complete pinned environment is [requirements-mujoco.lock](../../requirements-mujoco.lock).
These are the versions used here, not a claim of compatibility with every GPU.

```bash
# From the repository root, with uv already installed:
OPTIQ_VENV="$PWD/.venv-v2" bash scripts/setup_mujoco_env.sh
export OPTIQ_PYTHON="$PWD/.venv-v2/bin/python"
cp .env.example .env
# Edit .env to supply WANDB_API_KEY. Do not commit credentials.
export OPTIQ_ENV_FILE="$PWD/.env"

# Read-only configuration inspection; no W&B run and no training:
scripts/run_v2.sh --list
scripts/run_v2.sh 0 --check
"$OPTIQ_PYTHON" -c 'import jax; print(jax.devices())'
```

The setup script syncs the designated venv. Select a new isolated venv when the
machine also runs other projects. Rendering videos is not required for headless
training; the launcher sets MUJOCO_GL=egl. `.env.example` has no secret values.
The actual experiment runner requires a JAX GPU backend and online W&B logging.
Tests and configuration inspection do not require W&B credentials.

## 2. Train the selected algorithm

```bash
# One GPU per process. Run these in separate managed workers/terminals:
CUDA_VISIBLE_DEVICES=0 scripts/run_v2.sh 0 progress_bar=false
CUDA_VISIBLE_DEVICES=1 scripts/run_v2.sh 1 progress_bar=false
CUDA_VISIBLE_DEVICES=2 scripts/run_v2.sh 2 progress_bar=false
CUDA_VISIBLE_DEVICES=3 scripts/run_v2.sh 3 progress_bar=false
```

On a supervisor-based instance, [supervisor_v2.sh](../../scripts/supervisor_v2.sh)
is the environment/logging wrapper; pass its seed argument from a managed service.
For new installations, set `OPTIQ_ROOT`, `OPTIQ_PYTHON`, `OPTIQ_ENV_FILE` and
`CUDA_VISIBLE_DEVICES` explicitly in the service environment. Existing deployment
files can contain this instance's absolute paths: adapt those before installing them.
Do not run two workers on the same GPU unintentionally.

The launcher uses a repo-local `.env` first. On this instance it retains the
existing `/workspace/OptiQ-heechan-no-anchor/.env` fallback. An explicitly exported
`OPTIQ_ENV_FILE` takes precedence. W&B defaults to entity `OptiQ`, project
`optiq_mujoco_v2_confirmation`; use the existing account credentials. For an
independent implementer without access to that team:

```bash
WANDB_ENTITY=your-entity CUDA_VISIBLE_DEVICES=0 scripts/run_v2.sh 0 \
  wandb.project=your-v2-project progress_bar=false
```

The config, not `WANDB_PROJECT`, sets the final project's name; a Hydra override
is explicit. Every run receives a UTC timestamp and random suffix, its own output
directory and W&B run. No existing run is resumed or overwritten by default.

Other MuJoCo environments use the same algorithm and environment-specific spaces:

```bash
scripts/run_v2.sh 0 --check benchmark=ant
scripts/run_v2.sh 0 --check benchmark=halfcheetah
scripts/run_v2.sh 0 --check benchmark=walker2d
scripts/run_v2.sh 0 --check benchmark=hopper
# Remove --check only when intending to train.
```

Humanoid is the benchmark with the reported final four-seed confirmation.
Configuration availability for other tasks is not evidence of the same performance.
Changing T, teacher aggregation, σ floor, OT epsilon, latent prior or guard behavior
creates a new experiment and should be labeled separately.

## 3. Independent implementation

Implement [PSEUDOCODE.md](PSEUDOCODE.md) in full. The minimum components are:
semi-implicit actor, scalar twin critics and critic targets, uniform replay,
joint tanh-mixture log density, conditional teacher sampling, self-normalized IS,
batched log-domain Sinkhorn, full-row NLL, state-batched soft guard and Adam rollback.
All arrays are float32 in training; the NumPy reference may calculate in float64
for cross-checking. No diffusion package is conceptually needed for this algorithm.

Recommended port validation sequence:

1. Run [reference_math.py](reference_math.py) with NumPy alone.
2. Feed identical network outputs and base random draws to the reference and port;
   compare densities, floored proposals, weights, Sinkhorn, NLL and analytic gradients.
3. Test terminal and time-limit TD targets separately. Verify the twin loss SUM and
   joint critic gradient clipping. Preserve full actor optimizer rollback.
4. Run short Humanoid integration checks, including checkpoint restoration.
5. Run seeds 0/1/2/3 with the full parameters and evaluation protocol.

Sampling distribution parity is achievable across frameworks; exact random stream
and floating-point parity is not implied by using the same integer seed. PyTorch's
default Linear initialization and exact GELU differ from this implementation's
specified choices. Make these differences explicit in a replication report.

## 4. Review commands

```bash
"$OPTIQ_PYTHON" docs/v2/reference_math.py
JAX_PLATFORMS=cpu OPTIQ_TEST_WANDB=0 "$OPTIQ_PYTHON" -m pytest -q \
  tests/test_v2_final.py tests/test_conditional_proposal.py \
  tests/test_distributional_distillation.py tests/test_soft_guard.py \
  tests/test_soft_improvement.py tests/test_semi_implicit.py \
  tests/test_grad_clip_standalone.py tests/test_scalar_critic.py
"$OPTIQ_PYTHON" scripts/verify_v2_final.py
```

The tests compare archived configurations against the pre-reorganization snapshot,
and final algorithm parameters against the completed run's frozen values.
They include an actual short Humanoid loop with 256×3 networks, M16, 16×64 OT,
J8/BG32 guard, time-limit transitions and checkpoint restoration. Shortened warmup
and training batch in that integration test are test-only settings, not new defaults.

## 5. Evaluation and data preservation

During training, evaluate at environment step 1 and every 5000 steps, 10 stochastic
episodes each. Keep all 21 evaluations from 900K through 1M for each seed, then
average over four seeds. Use the final actor checkpoint, not the best checkpoint,
for the independent 50-episode evaluation. Its environment seeds are
1100000..1100049; policy seeds are 1110000..1110049. Count all termination types.
Detailed callback RNG rules and evaluation/checkpoint order are in the pseudocode.

Original artifacts remain under `outputs/v2_confirmation/` and
`outputs/v2_improvement/`. The latter contains comparison manifests, final
independent episode records and source/checkpoint hashes. The W&B project is
[OptiQ / optiq_mujoco_v2_confirmation](https://wandb.ai/OptiQ/optiq_mujoco_v2_confirmation).
Path cleanup does not move or delete those artifacts, nor change running jobs.

Actor/critic checkpoints are suitable for evaluation. Exact mid-training resume
also requires replay, environment, RNG and counter state, which the historical
checkpoint format does not completely store. Do not describe a checkpoint-only
restart as an uninterrupted reproduction.

## 6. Archived variants

See [the archive index](../archive/v2/README.md). The supplied original algorithm
is selected explicitly with `OPTIQ_CONFIG=archive/v2/original`. Final `mujoco_v2`
does not inherit that original configuration. Other historical config names remain
compatibility aliases. Shared diagnostic/monitor scripts retain their existing paths
because archived manifests and running services reference them.
