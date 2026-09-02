#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  exit 2
fi

task=$1
case "$task" in
  run|trot|walk|stand) ;;
  *) echo "unknown dog task: $task" >&2; exit 2 ;;
esac

completion_dir=/workspace/DIME/outputs/optiq_dime_dog/completed
mkdir -p "$completion_dir"

# Do not compete with the active T=0.25 worker for the task's assigned GPU.
baseline_marker="$completion_dir/${task}_seed1.done"
while [[ ! -f "$baseline_marker" ]]; do
  echo "Waiting for T=0.25 task=$task seed=1 to finish"
  sleep 60
done

for temperature in 0.5 0.1; do
  temperature_tag=${temperature/./p}
  for seed in 1; do
    marker="$completion_dir/${task}_T${temperature_tag}_seed${seed}.done"
    in_progress_marker="$completion_dir/${task}_T${temperature_tag}_seed${seed}.in_progress"
    if [[ -f "$marker" ]]; then
      echo "Skipping completed task=$task temperature=$temperature seed=$seed"
      continue
    fi
    if [[ -f "$in_progress_marker" ]]; then
      echo "Skipping externally running task=$task temperature=$temperature seed=$seed"
      continue
    fi
    echo "Starting task=$task temperature=$temperature seed=$seed on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    /workspace/DIME/scripts/run_dog_experiment.sh \
      "$task" "$seed" 1000000 "$temperature"
    touch "$marker"
  done
done
