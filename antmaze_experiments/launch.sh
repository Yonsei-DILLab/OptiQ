#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/heechan/.venv-ddiffpg-native/bin:$PATH
export LD_LIBRARY_PATH=/home/heechan/.mujoco/mujoco210/bin
export MUJOCO_PY_FORCE_CPU=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export WANDB_PROJECT=gmm-trg WANDB_ENTITY=OptiQ
exec /home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.run "$@"
