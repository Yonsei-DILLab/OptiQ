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
temperature=0.25
proposal_std=0.2
proposal_clip=0.5
minimum_source_ess=${MINIMUM_SOURCE_ESS:-16}
ess_tag=${minimum_source_ess//./p}
output_root=outputs/optiq_dime_dog_adaptive_beta_ess${ess_tag}_t0p25_sigma0p2_clip2p5
completion_dir=/workspace/DIME/${output_root}/completed
assignment_dir=/workspace/DIME/${output_root}/assignments
lock_dir=/workspace/DIME/${output_root}/locks
mkdir -p "$completion_dir" "$assignment_dir" "$lock_dir"

assignment_file="$assignment_dir/${task}.gpu"
if [[ -f "$assignment_file" ]]; then
  assigned_gpu=$(<"$assignment_file")
  if [[ "${CUDA_VISIBLE_DEVICES:-}" != "$assigned_gpu" ]]; then
    echo "Skipping task=$task on GPU=${CUDA_VISIBLE_DEVICES:-unset}; assigned GPU=$assigned_gpu"
    exit 0
  fi
fi

exec 9>"$lock_dir/${task}.lock"
if ! flock -n 9; then
  echo "Skipping duplicate task=$task; another process holds its run lock"
  exit 0
fi

source /workspace/.venv-dime/bin/activate
cd /workspace/DIME
export XLA_PYTHON_CLIENT_PREALLOCATE=false

marker="$completion_dir/${task}_seed${seed}.done"
if [[ -f "$marker" ]]; then
  echo "Skipping completed task=$task seed=$seed"
  exit 0
fi

run_suffix=${RUN_SUFFIX:-}
if [[ -n "$run_suffix" ]]; then
  run_suffix="_${run_suffix}"
fi
run_name="optiq_dime_dog_${task}_seed${seed}_adaptiveBeta_ESS${ess_tag}_N16R5_T0p25_sigma0p2_clip0p5_1m${run_suffix}"
echo "Starting adaptive-beta task=$task ESS>=$minimum_source_ess seed=$seed on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python run_optiq_dime.py \
  task="$task" \
  seed="$seed" \
  total_steps=1000000 \
  checkpoint_interval=50000 \
  alg.actor.temperature="$temperature" \
  alg.actor.proposal_std="$proposal_std" \
  alg.actor.proposal_clip="$proposal_clip" \
  alg.actor.density_beta=1.0 \
  alg.actor.adaptive_density_beta=true \
  alg.actor.minimum_source_ess="$minimum_source_ess" \
  output_root="$output_root" \
  run_name="$run_name" \
  wandb.project="optiq_dime_dog_adaptive_beta_ess${ess_tag}_t0p25_sigma0p2_clip2p5_1seed_1m" \
  wandb.group="dog-${task}_adaptive-beta-ess${ess_tag}-sigma0p2-clip2p5"
touch "$marker"
