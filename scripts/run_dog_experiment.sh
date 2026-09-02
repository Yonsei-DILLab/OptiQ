#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 TASK SEED [TOTAL_STEPS]" >&2
  exit 2
fi

task=$1
seed=$2
total_steps=${3:-1000000}
case "$task" in
  run|trot|walk|stand) ;;
  *) echo "unknown dog task: $task" >&2; exit 2 ;;
esac

source /workspace/.venv-dime/bin/activate
cd /workspace/DIME
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec python run_optiq_dime.py \
  task="$task" \
  seed="$seed" \
  total_steps="$total_steps"
