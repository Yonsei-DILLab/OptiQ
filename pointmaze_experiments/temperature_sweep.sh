#!/bin/bash
set -euo pipefail
gpu=$1
temperature=$2
phase=${3:-main}
[[ "$temperature" == 3 || "$temperature" == 5 ]]
export CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg MUJOCO_GL=egl USE_FLAX=0 USE_TORCH=1 PYTHONDONTWRITEBYTECODE=1
export MUJOCO_PY_MUJOCO_PATH=/workspace/antmaze-temperature-20260924/mujoco210
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
read -r WANDB_API_KEY < /workspace/optiq-clean-seed3to7-20260909/wandb_api_key
export WANDB_API_KEY
cd /workspace/optiq-drac-temperature-20260926
exec 9>"/tmp/ibolt-drac-pointmaze-gpu${gpu}.lock"
flock -n 9
python=/workspace/antmaze-temperature-20260924/venv/bin/python
common=(--source /workspace/drac-official-20260926 --seed 0 --temperature "$temperature" --robustness)
if [[ "$phase" == preflight ]]; then
 exec "$python" -u -m pointmaze_experiments.train "${common[@]}" --maze simple \
  --steps 272 --warmup 256 --eval-every 272 --eval-episodes 2 --preflight \
  --output "/workspace/ibolt-drac-T${temperature}-preflight-20260926"
fi
failures=0
for maze in simple medium hard; do
 case "$maze" in simple) steps=100000;; medium) steps=200000;; hard) steps=300000;; esac
 output="/workspace/ibolt-drac-${maze}-T${temperature}-s0-20260926"
 # Fresh-only: never overwrite or silently resume an old result.
 if [[ -e "$output" ]]; then
  echo "Refusing existing output: $output"; failures=$((failures+1)); continue
 fi
 if "$python" -u -m pointmaze_experiments.train "${common[@]}" --maze "$maze" \
    --steps "$steps" --output "$output"; then
  echo "COMPLETED $maze T=$temperature"
 else
  echo "FAILED $maze T=$temperature; preserving artifacts; advancing independent job"
  failures=$((failures+1))
 fi
done
exit "$failures"
