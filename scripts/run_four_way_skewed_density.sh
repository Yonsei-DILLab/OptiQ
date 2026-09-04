#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SEED TEMPERATURE LABEL" >&2
  exit 2
fi

seed=$1
temperature=$2
label=$3
output_root="outputs/four_way_multigoal/skewed_density_gpu_${label}"
completion_dir="/workspace/DIME-skewed-proposal/${output_root}/completed"
lock_dir="/workspace/DIME-skewed-proposal/${output_root}/locks"
mkdir -p "$completion_dir" "$lock_dir"

exec 9>"$lock_dir/seed_${seed}.lock"
if ! flock -n 9; then
  echo "Skipping duplicate seed=${seed} label=${label}; lock is held"
  exit 0
fi

marker="$completion_dir/seed_${seed}.done"
if [[ -f "$marker" ]]; then
  echo "Skipping completed seed=${seed} label=${label}"
  exit 0
fi

source /workspace/.venv-dime/bin/activate
cd /workspace/DIME-skewed-proposal
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m experiments.four_way_multigoal.train_skewed \
  --seed "$seed" \
  --total-steps 100000 \
  --warmup-steps 1000 \
  --proposal-mode skewed \
  --proposal-std 0.2 \
  --proposal-clip 0.5 \
  --no-include-anchor \
  --temperature "$temperature" \
  --log-interval 5000 \
  --output-dir "$output_root"

touch "$marker"
