# Rebuttal experiment environment

Base: `codex/ibolt-anonymous`, commit `1c8d7a453d4de4c79ee60844ffdf8d0140f109a8`.

The preparation hosts each have four RTX 5090 GPUs (32 GB per GPU).
Use the separate Blackwell profile with Python 3.11, JAX 0.6.2,
Flax 0.10.6 and PyTorch 2.8.0 with CUDA 12.8. The original
`requirements.txt`, algorithm and benchmark configurations are preserved.
Changing the numerical stack can change results; record package versions and
the Git revision for every experiment. Preparation checks are execution checks,
not completed benchmark experiments.

JAX moved its build to CUDA 12.8 in 0.6.0:
https://github.com/jax-ml/jax/blob/jax-v0.6.2/CHANGELOG.md
PyTorch's official CUDA 12.8 wheels are documented at:
https://pytorch.org/get-started/previous-versions/

## Setup and validation

From the checkout root (`/workspace/iBOLT` on the preparation hosts):

```bash
bash scripts/setup-blackwell.sh
source scripts/activate-experiments.sh
python scripts/check-gpu.py
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests
CUDA_VISIBLE_DEVICES=0 python train.py benchmark=hopper wandb.activate=false \
  total_steps=6 alg.learning_starts=2 alg.actor.learning_starts=2 \
  alg.batch_size=4 alg.buffer_size=32 num_eval_episodes=1 \
  eval_interval=6 diagnostic_interval=6 checkpoint_interval=6
```

GPU validation executes matrix multiplication through JAX and PyTorch on every
visible GPU. The training smoke test retains `require_gpu=true`.
The setup uses an isolated `.venv`; system Python and the default Vast environment
are not modified. `uv` and network access to PyPI, GitHub and the official
PyTorch wheel index are required. A successful installation is captured in
`requirements-blackwell.lock.txt` for repeat installations.

## Starting experiments

```bash
cd /workspace/iBOLT
source scripts/activate-experiments.sh
CUDA_VISIBLE_DEVICES=0 python train.py benchmark=hopper seed=0
CUDA_VISIBLE_DEVICES=1 python train.py benchmark=walker2d seed=0
CUDA_VISIBLE_DEVICES=2 python train.py benchmark=halfcheetah seed=0
CUDA_VISIBLE_DEVICES=3 python train.py benchmark=ant seed=0
```

These are separate foreground commands; launch each in its own session or a
managed job. Each run uses one GPU. Choose task and seed allocation explicitly;
no full experiment is started by setup. W&B defaults to offline. See `README.md`,
`gmm40/PROTOCOL.md` and `pointmaze/PROTOCOL.md` for the respective protocols.

Keep generated outputs, credentials, logs and virtual environments out of Git.
Neither preparation host has a persistent workspace volume: stop/start retains
the container, but recycle/destroy removes its filesystem. Copy experiment
outputs off the host before recycling or destroying it.
