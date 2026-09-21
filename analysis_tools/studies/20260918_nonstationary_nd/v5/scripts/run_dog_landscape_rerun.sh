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
proposal_std=0.1
proposal_clip=0.15
output_root=outputs/optiq_dime_dog_truncation_k1p5_t0p25_sigma0p1_landscape_rerun
completion_dir=/workspace/DIME/${output_root}/completed
mkdir -p "$completion_dir"

source /workspace/.venv-dime/bin/activate
cd /workspace/DIME
export XLA_PYTHON_CLIENT_PREALLOCATE=false

marker="$completion_dir/${task}_seed${seed}.done"
if [[ -f "$marker" ]]; then
  echo "Skipping completed task=$task seed=$seed"
  exit 0
fi

run_name="optiq_dime_dog_${task}_seed${seed}_N16R5_argmax_T0p25_sigma0p1_clip0p15_trunc1p5_landscape_rerun_1m"
echo "Starting landscape rerun task=$task temperature=$temperature sigma=$proposal_std clip=$proposal_clip seed=$seed on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python run_optiq_dime.py \
  --config-name=optiq_dime_dog \
  task="$task" \
  seed="$seed" \
  total_steps=1000000 \
  checkpoint_interval=50000 \
  alg.actor.temperature="$temperature" \
  alg.actor.proposal_std="$proposal_std" \
  alg.actor.proposal_clip="$proposal_clip" \
  output_root="$output_root" \
  run_name="$run_name" \
  wandb.group="dog-${task}_optiq-dime-T0p25-sigma0p1-trunc1p5-landscape-rerun"
touch "$marker"
