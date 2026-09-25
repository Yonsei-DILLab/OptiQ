#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg MUJOCO_GL=egl USE_FLAX=0 USE_TORCH=1
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_PY_MUJOCO_PATH=/workspace/antmaze-temperature-20260924/mujoco210
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
read -r WANDB_API_KEY < /workspace/optiq-clean-seed3to7-20260909/wandb_api_key
export WANDB_API_KEY
cd /workspace/optiq-drac-pointmaze-20260926
exec 9>/tmp/ibolt-drac-pointmaze-gpu0.lock
flock -n 9
exec /workspace/antmaze-temperature-20260924/venv/bin/python -u -m pointmaze_experiments.train \
 --source /workspace/drac-official-20260926 --maze simple --seed 0 --temperature 1 \
 --output /workspace/ibolt-drac-pointmaze-simple-T1-s0-20260926
