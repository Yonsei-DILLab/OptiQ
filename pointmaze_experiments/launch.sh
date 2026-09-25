#!/bin/bash
set -euo pipefail
temperature=${1:?};gpu=${2:?};output=${3:?};shift 3
export CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=online USE_FLAX=0 USE_TORCH=1
read -r WANDB_API_KEY < /workspace/optiq-clean-seed3to7-20260909/wandb_api_key || test -n "${WANDB_API_KEY:-}"
export WANDB_API_KEY
cd /workspace/optiq-pointmaze-20260926
exec flock -n "/tmp/pointmaze-gpu${gpu}.lock" /workspace/antmaze-temperature-20260924/venv/bin/python -u -m pointmaze_experiments.run --temperature "$temperature" --output "$output" "$@"
