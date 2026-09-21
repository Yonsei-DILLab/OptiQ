#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TASK TEMPERATURE" >&2
  exit 2
fi

task=$1
temperature=$2
case "$task" in
  run|trot|walk|stand) ;;
  *) echo "unknown dog task: $task" >&2; exit 2 ;;
esac
case "$temperature" in
  0.1|0.5) ;;
  *) echo "unsupported temperature: $temperature" >&2; exit 2 ;;
esac

seed=1
completion_dir=/workspace/DIME/outputs/optiq_dime_dog/completed
temperature_tag=${temperature/./p}
marker="$completion_dir/${task}_T${temperature_tag}_seed${seed}.done"
in_progress_marker="$completion_dir/${task}_T${temperature_tag}_seed${seed}.in_progress"
mkdir -p "$completion_dir"

if [[ -f "$marker" ]]; then
  echo "Skipping completed task=$task temperature=$temperature seed=$seed"
  exit 0
fi

cleanup() {
  rm -f "$in_progress_marker"
}
trap cleanup EXIT
touch "$in_progress_marker"

echo "Starting early sweep task=$task temperature=$temperature seed=$seed on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
/workspace/DIME/scripts/run_dog_experiment.sh \
  "$task" "$seed" 1000000 "$temperature"
touch "$marker"
