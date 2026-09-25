#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/home/heechan/.mujoco/mujoco210/bin
export MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1 USE_FLAX=0 USE_TORCH=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg WANDB_MODE=disabled
exec /home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.center_v1_probe \
  --source /home/heechan/OptiQ-single-v1-20260925 \
  --run /home/heechan/optiq-experiments/antmaze-v1-single-utd1-100k-20260925/run \
  --turning-reference /home/heechan/optiq-experiments/antmaze-v1-center-diagnostic-20260925 \
  --output /home/heechan/optiq-experiments/antmaze-v1-turning-diagnostic-20260925
