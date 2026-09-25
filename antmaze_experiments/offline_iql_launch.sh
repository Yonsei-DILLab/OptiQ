#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/heechan/.venv-ddiffpg-native/bin:$PATH
export LD_LIBRARY_PATH=/home/heechan/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/home/heechan/.mujoco/mujoco210
export MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export USE_FLAX=0 USE_TORCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg PYTHONUNBUFFERED=1
exec /home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.offline_iql \
    --output /home/heechan/optiq-experiments/antmaze-v1-uniform1M-iql-20260925
