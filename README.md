# iBOLT

Anonymous review snapshot of the MuJoCo implementation. This repository is
self-contained: no private experiment folders, cluster paths, or W&B account
are required. It contains implementation and reproduction settings, not
pretrained checkpoints or benchmark results.

## Installation

Use Python 3.11 on Linux with an NVIDIA GPU:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install 'jax[cuda12]==0.4.33'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
```

The pinned JAX/CUDA stack matches the training implementation. GPU architecture
and driver compatibility must be checked separately (particularly newer GPU
architectures). CPU validation can use the base requirements without CUDA.

## Train

```bash
python train.py benchmark=hopper seed=0
python train.py benchmark=walker2d seed=0
python train.py benchmark=halfcheetah seed=0
python train.py benchmark=ant seed=0
python train.py benchmark=humanoid seed=0
```

Each command runs 1M interactions, including 5K random-action warmup. Repeat
with `seed=0` through `seed=4` for five independent runs. W&B defaults to offline
mode; local CSV, evaluation arrays, configuration and checkpoints are written
under `outputs/`. To disable W&B entirely, use `wandb.activate=false`. For online
logging, run `wandb login` and pass `wandb.mode=online wandb.entity=YOUR_ENTITY`.
Never add credentials or generated output to this repository.

| Environment | Temperature | Exploration multiplier |
|---|---:|---:|
| Hopper-v4 | 0.05 | 0.10 |
| Walker2d-v4 | 0.10 | 0.10 |
| HalfCheetah-v4 | 0.25 | 0.15 |
| Ant-v4 | 0.25 | 0.10 |
| Humanoid-v4 | 0.10 | 0.15 |

Shared defaults: two 256-unit hidden layers in actor and critics; GELU;
twin scalar critics with plain TD; batch 256; UTD 1; Adam learning rates
3e-4; discount 0.99; target update 0.005; replay capacity 1M; no gradient
clipping; N=M=64; full density correction; trainable log-scale in [-5,-1],
initialized at -1.

## Algorithm and evaluation

For a state, Gaussian latent inputs produce conditional centers and scales.
Centers are tanh-bounded. Conditional distributions are **box-truncated
Gaussians on [-1,1]**, not tanh-squashed Gaussian densities. Candidate actions
are sampled from the finite conditional mixture. Their value weights include
proposal-density correction. The actor maximizes weighted **marginal mixture
likelihood**; this version does not use OT, hard assignments or conditional-row
NLL. The preserved internal `optiq_dime` module name reflects implementation
ancestry, not an extra diffusion generation stage. Legacy helpers remain as
dependencies; the public entry point selects only direct mixture likelihood.

Both evaluation modes are logged every 5K steps, with 10 episodes per mode:
`eval/zero_z/mean_reward` uses z=0 and the center action;
`eval/stochastic_z/mean_reward` draws z independently at each action and also
uses only the center. Neither adds conditional Gaussian or external noise.
Do not combine these metrics when aggregating results.

DACER-inspired behavior exploration adds `c * alpha * Normal(0,I)` to the
collected action, followed by clipping to [-1,1]. The executed action is stored
in replay. This extra noise is absent from teacher sampling, TD targets and
evaluation. Alpha starts at 0.27, with log-alpha Adam learning rate 0.03,
target entropy -0.9 times action dimension, and updates every 10K learner
updates. The entropy proxy fits a three-component GMM to 200 current-policy
actions at replay states. This proxy is not exact mixture entropy.

## Ablations

```bash
# Temperature, without changing other settings
python train.py benchmark=ant temperature=0.5 seed=0
# No external behavior exploration
python train.py benchmark=ant dacer.enabled=false seed=0
# N=M=16 (M=N * proposals_per_policy_sample)
python train.py benchmark=ant alg.actor.num_policy_samples=16 seed=0
# Constant conditional scale; mu remains trainable
python train.py benchmark=ant fixed_log_std=-3 seed=0
```

Fixed log-scales -1, -3, -5 correspond to sigma approximately 0.368, 0.0498,
0.00674. This is a constant output, not a frozen sigma head that can drift
through shared-trunk updates.

## Validation

```bash
pytest -q tests
python train.py benchmark=hopper require_gpu=false wandb.activate=false \
  total_steps=6 alg.learning_starts=2 alg.actor.learning_starts=2 \
  alg.batch_size=4 alg.buffer_size=32 num_eval_episodes=1 \
  eval_interval=6 diagnostic_interval=6 checkpoint_interval=6
```

The short command is an execution check, not a performance experiment.
Tests cover truncated density normalization, sampling, likelihood gradients,
and a short learner update in each of the five environments.

## Layout and attribution

- `train.py`, `environment.py`: training, logging and evaluation setup.
- `configs/`: task defaults and ablation overrides.
- `optiq_dime/`: actor, density, weighted likelihood and critic updates.
- `exploration.py`: behavior-only entropy-regulated exploration.
- `common/`, `models/`, `diffusion/`: inherited training infrastructure.
- `tests/`: numerical and integration checks.

Inherited DIME infrastructure is retained under its MIT license; see `LICENSE`.
Stable-Baselines3 supplies the environment/replay interfaces. The exploration
regulator is inspired by DACER (arXiv:2405.15177); it is a behavior-only adaptation,
not an assertion of identical DACER training semantics. Third-party attribution
is retained for license compliance.

For anonymous review, publish the selected branch through the anonymization
service, not the identifying upstream repository URL. Exclude outputs and
runtime logs from the mirror, and inspect its preview before sharing.
