#!/bin/bash
set -euo pipefail
task=${1:?};gpu=${2:?}
export CUDA_VISIBLE_DEVICES="$gpu" CAMPAIGN_GPU="$gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_PY_MUJOCO_PATH=/workspace/antmaze-temperature-20260924/mujoco210
export LD_LIBRARY_PATH=/workspace/antmaze-temperature-20260924/mujoco210/bin:/usr/lib/nvidia
export USE_FLAX=0 USE_TORCH=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=online OPTIQ_WANDB_PROJECT=jaehun-antmaze OPTIQ_CAMPAIGN=ibolt-antmaze-t3-single-1m-20260926
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
read -r WANDB_API_KEY < /workspace/optiq-clean-seed3to7-20260909/wandb_api_key || test -n "${WANDB_API_KEY:-}"
export WANDB_API_KEY
cd /workspace/optiq-antmaze-t3-1m-20260926
exec 9>"/tmp/ibolt-antmaze-t3-gpu${gpu}.lock"
flock -n 9
args=(--method optiq --task "$task" --temperature 3 --budget-steps 1000000
 --reward-profile dense --noveld off --dacer off --eval-starts fixed
 --collection-profile single-update1 --optiq-config-profile basic
 --eval-interval 25000 --interim-eval-episodes 100 --final-eval-episodes 100
 --save-intermediate-policy)
runtime=/workspace/antmaze-temperature-20260924/venv/bin/python
"$runtime" -u -m antmaze_experiments.run "${args[@]}" --preflight --output "/workspace/ibolt-antmaze-T3-${task}-20260926-preflight"
exec "$runtime" -u -m antmaze_experiments.run "${args[@]}" --output "/workspace/ibolt-antmaze-T3-${task}-20260926-s0"
