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

seed=1
temperature=0.01
output_root=outputs/optiq_dime_dog_truncated_density_t0p01_n16r5_anchor_1m
completion_dir=/workspace/DIME-skewed-proposal/${output_root}/completed
lock_dir=/workspace/DIME-skewed-proposal/${output_root}/locks
mkdir -p "$completion_dir" "$lock_dir"

exec 9>"$lock_dir/${task}.lock"
if ! flock -n 9; then
  echo "Skipping duplicate task=$task; another process holds its run lock"
  exit 0
fi

marker="$completion_dir/${task}_seed${seed}.done"
if [[ -f "$marker" ]]; then
  echo "Skipping completed task=$task seed=$seed"
  exit 0
fi

source /workspace/.venv-dime/bin/activate
cd /workspace/DIME-skewed-proposal
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_name="optiq_dime_dog_${task}_seed${seed}_truncated_T0p01_densitycorr_anchor_N16R5_1m"
python run_optiq_dime.py \
  task="$task" \
  seed="$seed" \
  total_steps=1000000 \
  checkpoint_interval=50000 \
  alg.actor.proposal_sampling_mode=stratified \
  alg.actor.temperature="$temperature" \
  alg.actor.proposal_std=0.2 \
  alg.actor.proposal_clip=0.5 \
  alg.actor.include_anchor=true \
  alg.actor.density_correction=true \
  alg.actor.density_beta=1.0 \
  alg.actor.adaptive_density_beta=false \
  output_root="$output_root" \
  run_name="$run_name" \
  wandb.project=optiq_dime_dog_truncated_density_t0p01_n16r5_anchor_1seed_1m \
  wandb.group="dog-${task}_truncated-density-t0p01-anchor"

touch "$marker"
