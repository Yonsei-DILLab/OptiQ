# No-anchor OptiQ + DIME baseline

Branch: `critic-dime-no-anchor`, created from `heechan` at `c6132fd`.
The DIME critic lineage comes from `critic-dime` at `8b8fee1`.
The separate worktree on the experiment server is
`/workspace/OptiQ-critic-dime-no-anchor`.

## Algorithm

For each replay state:

1. Draw 16 actor actions and use them as the centers of a uniform KDE.
2. Define all 16 truncated Gaussian kernels before drawing candidates. Each
   kernel uses sigma 0.2, local displacement bounds ±0.5, and normalized action
   bounds [-1, 1]. Sampling uses inverse CDF truncation.
3. Draw four random actions per kernel, giving 64 candidates. The policy centers
   are not inserted into the candidate set (`include_anchor=false`). This is
   stratification over mixture components, with equal allocation to each.
4. Evaluate every candidate under the same full 16-component KDE. No KDE is
   fitted to the newly generated 64-candidate cloud.
5. With the mean of the twin critics, form candidate weights
   `softmax(Q / 0.25 - 0.1 * log KDE_density)`.
6. Solve an entropic OT problem with 16 uniform actor rows and 64 weighted
   candidate columns. The cost/transport tensor has shape `[batch, 16, 64]`;
   batch size is 256. Sinkhorn epsilon is 0.05 with 30 iterations, using the
   inherited mean-normalized squared action distance.
7. Select each row's argmax transport target and apply the inherited pointwise
   MSE (mean over samples of the sum of squared action-coordinate errors).

“No anchor” removes deterministic candidate slots. The KDE still needs the 16
actor samples as kernel centers and OT rows. Sampling is stratified, so the 64
draws are not IID draws with independently selected mixture components.

The inherited `heechan` sampler already evaluated the mixture defined by actor
centers, rather than fitting a KDE to the candidate cloud. The new shared KDE
object makes construction-before-sampling explicit and ties sampling and density
evaluation to the same distribution. At matching settings, regression tests
verify identical samples and matching densities. The baseline changes are the
disabled candidate anchors, the confirmed R=4, and the supplied configuration.

The distributional DIME critic, Batch Renorm, replay buffer, JAX/JIT update
path, and actor gradient objective remain the common implementation. Adaptive
beta is disabled. The inherited optional sampler/anchor modes remain available
for explicit future comparisons; the defaults above are the baseline.

## Configuration

The reference is recorded in `tests/data/no_anchor_reference.json`. Tests compare
every supplied algorithm parameter and evaluation setting against resolved
configuration. Two explicit user decisions override the pasted actor settings:
`include_anchor: true → false` and `proposals_per_policy_sample: 5 → 4`.

| Setting | Baseline |
| --- | --- |
| Single-run seed / launcher seeds | 0 / 0, 1, 2 |
| Training steps / JIT | 1,000,000 / enabled; GPU required by default |
| Warmup, critic and actor | 5,000 environment steps |
| Batch / replay / UTD / policy delay | 256 / 1,000,000 / 2 / 1 |
| Gamma / tau / policy tau | 0.99 / 1.0 / 1.0 |
| Critic | 2 × [2048, 2048], ReLU, 101 atoms, support [-3600, 3600] |
| Critic entropy coefficient | 0.005 |
| Critic dropout / layer norm / model resets | null / false / false |
| Batch normalization | enabled, momentum 0.99, `brn_actor`, warmup 100,000 |
| Critic / actor learning rate | 0.0003 / 0.0003 |
| Critic / actor Adam betas | (0.5, 0.999) / (0.9, 0.999) |
| Policy entropy coefficient | constant 0.0 |
| Actor hidden dimensions | [256, 256, 256] |
| N / R / anchors / sampler | 16 / 4 / false / stratified |
| Proposal sigma / clip | 0.2 / 0.5 in normalized action coordinates |
| Density correction / beta | enabled / fixed 0.1 |
| Source Q / reference | twin mean / uniform action |
| Temperature / Sinkhorn epsilon / iterations | 0.25 / 0.05 / 30 |
| Transport target / distillation | argmax / pointwise_mse |
| TD noise sigma / clip | 0.2 / 0.5 |
| Evaluation | every 5,000 steps, 10 episodes, stochastic |
| Initial evaluation | enabled, at step 1, matching the inherited callback |
| Detailed diagnostics / checkpoint interval | 5,000 / 50,000 steps |

The pasted `runtime` and `evaluation` entries map to the trainer's existing
top-level configuration keys. Actor `density_correction_beta` is the public
parameter; `density_beta` interpolates it for compatibility with the common
training implementation. Override the former; conflicting values are rejected.

| `benchmark` | Environment | Task | Success criterion |
| --- | --- | --- | --- |
| `pen_twirl_hard` (default) | `myoHandPenTwirlRandom-v0` | pen-twirl-hard | More than 5 solved steps per episode |
| `ant` | `Ant-v4` | ant | Return |
| `humanoid` | `Humanoid-v4` | humanoid | Return |
| `reach_hard` | `myoHandReachRandom-v0` | reach-hard | More than 5 solved steps per episode |
| `obj_hold_hard` | `myoHandObjHoldRandom-v0` | obj-hold-hard | More than 5 solved steps per episode |

MyoSuite 2.11.5 counts total solved steps, including nonconsecutive steps, and
uses strict `sum(solved) > successful_steps`. Evaluation logs success as a
fraction from 0 to 1 and separately saves the solved-step count per episode.
The registered maximum horizon is 50 for pen-twirl, 100 for reach, 75 for object
hold, and 1,000 for Ant and Humanoid. Actual horizons and action bounds are
captured in each run.

The new baseline uses the supplied critic support [-3600, 3600] for all five
benchmarks. The legacy `optiq_dime_mujoco` configuration still has [-1600, 1600].

## Environment and W&B

```bash
cd /workspace/OptiQ-critic-dime-no-anchor
bash scripts/setup_no_anchor_env.sh
cp -n .env.example .env
# Fill WANDB_API_KEY in .env and set WANDB_ENTITY if needed.
```

The setup script installs `requirements-no-anchor.lock` into the separate sibling
venv `/workspace/.venv-optiq-no-anchor`. It pins Python 3.11, JAX 0.4.33, Flax 0.9.0,
Gymnasium 0.29.1, SB3 2.1.0, MyoSuite 2.11.5, MuJoCo 3.3.0, and dm-control 1.0.28.
It requires `uv`; use `OPTIQ_VENV` to choose another installation path and export
`OPTIQ_PYTHON` when launching from a custom path.

This environment uses MuJoCo 3.3.0 to support MyoSuite. The running `heechan`
MuJoCo sweep uses its existing MuJoCo 2.3.7 environment. Comparing the two
branches' results therefore is not a controlled anchor-only ablation; matched
comparisons should use the same new environment and explicit matched settings.

W&B project defaults to `optiq_dime_no_anchor`. Shell variables take priority,
then `OPTIQ_ENV_FILE`, repo `.env`, and workspace `.env`. To reuse credentials
without copying them, export `OPTIQ_ENV_FILE=/workspace/OptiQ/.env` in the shell.
Explicit project/entity values in that file override configuration defaults.
Online W&B is required for the tracked experiment path. `.env` is ignored by Git.
On the current server, an independent ignored `.env` has already been populated
from the existing credentials, with owner-only permissions; the original file
was not changed. It can run without exporting `OPTIQ_ENV_FILE`.

## Launch

Preview the default three-run queue, or all nine benchmark/seed combinations:

```bash
bash scripts/run_no_anchor.sh --list
bash scripts/run_no_anchor.sh --list --benchmarks pen_twirl_hard,ant,humanoid
bash scripts/run_no_anchor.sh --list --benchmarks ant,humanoid
```

On an allocated GPU, queue MyoHand seeds 0, 1, 2 sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_no_anchor.sh
```

Run one task or partition the queue across workers:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_no_anchor.sh --task 0
CUDA_VISIBLE_DEVICES=0 bash scripts/run_no_anchor.sh --worker 0 3
CUDA_VISIBLE_DEVICES=1 bash scripts/run_no_anchor.sh --worker 1 3
CUDA_VISIBLE_DEVICES=2 bash scripts/run_no_anchor.sh --worker 2 3
```

Task IDs follow benchmark order, then seed order. `--seeds 0,1,2` is already the
default. `--benchmarks ant,humanoid` produces six runs. For long runs on the
Vast instance, use these foreground commands in a supervisor service, following
the instance guide. The worker lock prevents duplicate GPU workers within this
checkout; it does not coordinate other checkouts. Allocate GPUs before starting
workers, and use separate GPU indices for concurrent workers.

For reach-hard and object-hold-hard, the six-run queue is:

```bash
bash scripts/run_no_anchor.sh --list --benchmarks reach_hard,obj_hold_hard
```

`scripts/supervisor_no_anchor.conf.example` runs this queue across four GPUs:
GPU 0 runs reach seed 0 then hold seed 1; GPU 1 runs reach seed 1 then hold seed 2;
GPU 2 runs reach seed 2; GPU 3 runs hold seed 0. Each run uses the unchanged
no-anchor algorithm settings, 1,000,000 steps, and `successful_steps=5`.

The shared entry point also supports a single explicit run and configuration
inspection:

```bash
source /workspace/.venv-optiq-no-anchor/bin/activate
python run_optiq_dime.py --cfg job --resolve
CUDA_VISIBLE_DEVICES=0 python run_optiq_dime.py benchmark=humanoid seed=0
```

The inherited Dog scripts explicitly select `optiq_dime_dog`; the old MuJoCo
beta sweep explicitly selects `optiq_dime_mujoco`. Use `run_no_anchor.sh` for
this baseline's three-seed protocol.

## Outputs and diagnostics

Each run creates a unique directory under `outputs/optiq_dime_no_anchor/` with
resolved `config.json`, Git/package/device provenance, CSV/TensorBoard logs,
`evaluations.npz`, checkpoints, and a completion marker after successful training.
W&B receives scalar history and an artifact with configuration, evaluations,
and final actor/critic checkpoints at normal completion.

The evaluation NPZ is atomically replaced after each evaluation; MyoHand includes
`successes` and `solved_steps` alongside return and episode length. Local
periodic actor/critic checkpoints and already logged history remain if training
is interrupted. The inherited checkpoint format does not include replay/RNG
state for exact training resumption. The launcher fails on a failed task and
does not silently skip or automatically resume it.

Detailed metrics are logged at the diagnostic interval: full/Q-only/density-only
ESS, effective beta contribution, weight concentration, Q/logit dispersion,
policy spread, twin disagreement, and cross-critic selection gain. Anchor-relative
metrics are omitted when anchors are disabled. Core losses and evaluation metrics
retain the common training/evaluation logging path.

## Validation

The test suite covers the supplied configuration, queue seeds, KDE distribution
equivalence, actor/critic updates, actor warmup, diagnostics, MyoHand success
counting/persistence, and the inherited MuJoCo sampling/evaluation protocol.

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  /workspace/.venv-optiq-no-anchor/bin/python -m pytest -q
```

The initial 33-test suite passed on 2026-09-07 (43.88 seconds). The tests use the configured
online W&B credentials. All three real environments
were created and stepped in the isolated environment. A CPU/JIT MyoHand smoke
run used the full requested networks, batch size 256, N=16/R=4 and seed 0, with
12 steps, warmup 8, one evaluation episode every 4 steps, diagnostics every 4
steps, and checkpoint interval 10. It completed eight gradient updates, saved
evaluation/checkpoint artifacts, and uploaded to
[W&B run 035csnm0](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/035csnm0).
Additional CPU/JIT smoke runs for
[Ant](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/2v5ffyo0) and
[Humanoid](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/ysy6z764)
each completed eight updates with N=16/R=4 and seed 0. Those two checks used
eight environment steps, warmup 4, [32, 32] actor/critic networks, batch size 4,
replay capacity 32, evaluation/diagnostics interval 4, one evaluation episode,
and checkpoint interval 6. Evaluation arrays were finite, actual horizons were
recorded correctly, and both final actor/critic checkpoint files were present
for all three environments. The isolated lockfile passed `uv pip check` for
all 126 installed packages. This initial preparation used CPU/JIT while the
existing GPU workers were left running.
These shortened runs verify execution and logging, not learning performance.
No full baseline training queue was started during that initial preparation.

The reach-hard/object-hold-hard extension passed all six affected configuration
and queue tests. Both real environments reset and stepped successfully (reach:
115 observations, 39 actions, horizon 100; hold: 91 observations, 39 actions,
horizon 75). Full-network GPU/JIT smoke runs used 12 steps, warmup 8, batch size
256, N=16/R=4, seed 0, one evaluation episode every 4 steps, diagnostics every
4 steps, and checkpoint interval 10. Both completed eight gradient updates and
saved final checkpoints and evaluation artifacts:
[reach](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/6wtfg8qp),
[object hold](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/3muv1h3l).

On 2026-09-07, the user requested stopping the legacy sweep and launching these
two benchmarks. The old workers were interrupted and exited; their W&B histories,
evaluation NPZ files and periodic checkpoints were retained. The replacement
six-run queue uses the supervisor configuration above with all baseline training
and evaluation parameters unchanged.
