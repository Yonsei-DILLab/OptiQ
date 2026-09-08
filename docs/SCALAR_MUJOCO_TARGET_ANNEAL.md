# Scalar MuJoCo target-temperature experiments

The `heechan-no-anchor` branch adds a scalar critic path based on
[`mujoco-setting` at 7e2da67](https://github.com/Yonsei-DILLab/OptiQ/tree/7e2da67d2f0988f6f211b635f7311d32a0c8e8c6).
Its scalar network, no-BN state handling, target critic, twin-min bootstrap,
summed MSE, and algorithm settings are carried into the existing training loop.
The target-temperature schedule and diagnostics from this branch remain active.

## Critic setup

| Setting | Previous distributional run | Scalar run from mujoco-setting |
| --- | --- | --- |
| Twin critic hidden layers | 2048 x 2048 ReLU | 256 x 256 x 256 GELU |
| Outputs per critic | 101 categorical probabilities | One unbounded raw Q |
| Batch renormalization | Enabled | Disabled, no dummy BN computation |
| Updates per environment step | 2 | 1 |
| Next-state critic | Live CrossQ critic | Target critic |
| TD bootstrap aggregation | Twin mean distribution | Minimum of target Q1 and Q2 |
| Critic Polyak coefficient | 1.0 | 0.005 |
| Critic Adam betas | (0.5, 0.999) | (0.9, 0.999) |
| Critic objective | Categorical CE plus entropy term | Sum of per-critic MSE |
| Categorical support / entropy coefficient | [-1600, 1600] / 0.005 | None / 0 |

```text
y = stop_gradient(r + gamma * (1 - terminal) * min(Q1_target_next, Q2_target_next))
loss = mean((Q1 - y)^2) + mean((Q2 - y)^2)
target_params = 0.995 * target_params + 0.005 * updated_critic_params
```

Time-limit truncations retain the shared replay buffer's bootstrap handling.
The actor consumes raw scalar Q directly for weights and all training diagnostics.
`train/critic_td_rmse = sqrt(critic_loss / 2)` reports error averaged across
the two heads. Categorical entropy metrics are omitted for scalar runs.
Scalar checkpoints require the scalar network configuration; these experiments
start from scratch rather than loading distributional checkpoints.

This comparison changes the critic architecture and learning setup together.
It is not a controlled change of output representation alone.

## Retained experiment protocol

- Ant-v4 and HalfCheetah-v4, seed 0, 1,000,000 environment steps each.
- Target temperature `T(t) = 10 * (0.25 / 10) ** min(t / 400000, 1)`.
  Environment steps include the initial 5,000 random-action steps. Temperature
  reaches 0.25 at 400k and remains fixed; the UTD change does not change this clock.
- Density beta **0.1**, explicitly overriding mujoco-setting's beta **1.0**.
- The same 256 x 3 GELU actor, actor Adam (0.9, 0.999), learning rates 3e-4,
  actor target-copy coefficient 1.0, gamma 0.99, batch 256, replay 1M,
  actor/critic warmup 5k, policy delay 1, and entropy coefficient zero.
- No inserted anchors; KDE built from 16 policy samples, four stratified random
  candidates per component, 16 x 64 OT, proposal std 0.2 and clip 0.5.
- Sinkhorn epsilon 0.05, 30 iterations, argmax assignment, pointwise actor MSE,
  TD action noise std 0.2 / clip 0.5. Only the Q-weight temperature anneals.
- Evaluation at the start and every 5k with 10 stochastic episodes, diagnostics
  every 5k, checkpoints every 50k plus the first training step.

## Launch and logging

```bash
bash scripts/run_scalar_mujoco_target_anneal.sh --list
CUDA_VISIBLE_DEVICES=0 bash scripts/run_scalar_mujoco_target_anneal.sh --task 0
CUDA_VISIBLE_DEVICES=1 bash scripts/run_scalar_mujoco_target_anneal.sh --task 1
```

The launcher uses `/workspace/.venv-optiq-no-anchor` and the existing ignored
`.env`. Managed workers use `scripts/supervisor_mujoco_scalar.conf.example`.
Stop the previous `optiq-mujoco-target` workers before starting the scalar group;
Myo Reach workers on GPUs 2/3 are separate and continue unchanged.

The new config is `optiq_scalar_mujoco_target_anneal`. Its scalar algorithm config
can also be selected explicitly; legacy distributional defaults are retained.
Scalar support fields are null and incompatible type/atom/entropy settings fail
before training. Outputs use `outputs/optiq_scalar_mujoco_target_anneal` with
unique run directories. W&B retains the existing project and uses scalar tags
and groups, full resolved config, Git provenance, GPU, package versions, losses,
ESS, Q statistics, temperature, evaluations, and checkpoints.

Stopped MuJoCo outputs, W&B histories, evaluations, and previously saved
checkpoints stay at their original paths. Interruption does not promise a new
checkpoint at the exact stop step. Transition records are written to the ignored
`outputs/queue_manifests/` directory.

## Validation

`tests/test_scalar_critic.py` checks retained actor/evaluation configuration,
the scalar settings, target masking and stopped gradients, exact twin-min TD
updates against `mujoco-setting`, Polyak averaging, absence of BN, raw Q beyond
the old support, real Ant/HalfCheetah training, temperature and ESS diagnostics,
and checkpoint writes. The categorical update is also compared against the
pre-change implementation. Run the suite on CPU while training GPUs are busy:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /workspace/.venv-optiq-no-anchor/bin/python -m pytest -q tests
```

Validation uses the repository's existing online W&B fixture. Legacy Dog-specific
offline landscape scripts are not part of this scalar launch or validation path.
