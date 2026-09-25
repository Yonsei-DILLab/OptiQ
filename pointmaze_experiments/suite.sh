#!/bin/bash
set -euo pipefail
gpu=$1
maze=$2
phase=${3:-main}
export CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg MUJOCO_GL=egl USE_FLAX=0 USE_TORCH=1 PYTHONDONTWRITEBYTECODE=1
export MUJOCO_PY_MUJOCO_PATH=/workspace/antmaze-temperature-20260924/mujoco210
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
read -r WANDB_API_KEY < /workspace/optiq-clean-seed3to7-20260909/wandb_api_key
export WANDB_API_KEY
cd /workspace/optiq-drac-suite-20260926
exec 9>"/tmp/ibolt-drac-pointmaze-gpu${gpu}.lock"
flock -n 9
case "$maze" in simple) steps=100000;; medium) steps=200000;; hard) steps=300000;; *) exit 2;; esac
args=(--source /workspace/drac-official-20260926 --maze "$maze" --seed 0 --temperature 1 --steps "$steps" --robustness)
if [[ "$phase" == preflight ]]; then
 args+=(--steps 272 --warmup 256 --eval-every 272 --eval-episodes 2 --preflight --output "/workspace/ibolt-drac-suite-${maze}-preflight-20260926")
else
 args+=(--output "/workspace/ibolt-drac-suite-${maze}-T1-s0-20260926")
 if [[ "$maze" == simple ]]; then
  args+=(--checkpoint /workspace/ibolt-drac-pointmaze-simple-T1-s0-20260926/policy-0100000.msgpack)
 fi
fi
exec /workspace/antmaze-temperature-20260924/venv/bin/python -u -m pointmaze_experiments.train "${args[@]}"
