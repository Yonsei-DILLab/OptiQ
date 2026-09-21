#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 TASK SEED [TOTAL_STEPS] [TEMPERATURE]" >&2
  exit 2
fi

task=$1
seed=$2
total_steps=${3:-1000000}
temperature=${4:-0.25}
case "$task" in
  run|trot|walk|stand) ;;
  *) echo "unknown dog task: $task" >&2; exit 2 ;;
esac
case "$temperature" in
  0.1|0.25|0.5) ;;
  *) echo "unsupported OptiQ temperature: $temperature" >&2; exit 2 ;;
esac

single_seed_marker=/workspace/DIME/outputs/optiq_dime_dog/completed/.single_seed_only
if [[ "$seed" != "1" && -f "$single_seed_marker" ]]; then
  echo "Skipping task=$task temperature=$temperature seed=$seed: single-seed plan is active"
  exit 0
fi

temperature_tag=${temperature/./p}
extra_overrides=("alg.actor.temperature=$temperature")
if [[ "$temperature" != "0.25" ]]; then
  extra_overrides+=(
    "output_root=outputs/optiq_dime_dog_temperature_sweep"
    "run_name=optiq_dime_dog_${task}_seed${seed}_N16R5_argmax_T${temperature_tag}_1m"
    "wandb.group=dog-${task}_optiq-dime-T${temperature_tag}"
  )
fi

source /workspace/.venv-dime/bin/activate
cd /workspace/DIME
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec python run_optiq_dime.py \
  --config-name=optiq_dime_dog \
  task="$task" \
  seed="$seed" \
  total_steps="$total_steps" \
  "${extra_overrides[@]}"
