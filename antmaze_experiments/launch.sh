#!/usr/bin/env bash
set -euo pipefail
ANTMAZE_PYTHON=${ANTMAZE_PYTHON:-/home/heechan/.venv-ddiffpg-native/bin/python}
export PATH=$(dirname "$ANTMAZE_PYTHON"):$PATH
export LD_LIBRARY_PATH=${MUJOCO_PY_MUJOCO_PATH:-/home/heechan/.mujoco/mujoco210}/bin:${LD_LIBRARY_PATH:-}
export MUJOCO_PY_FORCE_CPU=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export USE_FLAX=0 USE_TORCH=1
export HF_HOME=${HF_HOME:-/home/heechan/OptiQ-ops/cache/huggingface}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export WANDB_PROJECT=antmaze WANDB_ENTITY=OptiQ
exec "$ANTMAZE_PYTHON" -m antmaze_experiments.run "$@"
