#!/usr/bin/env bash
#SBATCH --job-name=optiqM5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${OPTIQ_CONDA_ENV:-flowrl-ogbench}"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

task_id="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ -z "$task_id" || ! "$task_id" =~ ^[0-9]+$ || "$task_id" -gt 24 ]]; then
  echo "Usage: sbatch --array=0-24%8 $0" >&2
  echo "Local dry run: DRY_RUN=1 bash $0 TASK_ID" >&2
  exit 2
fi

env_keys=(hopper walker2d halfcheetah ant humanoid)
env_names=(Hopper-v4 Walker2d-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4)
env_index=$((task_id / 5))
seed=$((task_id % 5 + 1))
env_key="${env_keys[$env_index]}"
env_name="${env_names[$env_index]}"

project="${WANDB_PROJECT:-optiq_mujoco5}"
group="${env_name}"
run_name="optiq_${env_name}_seed${seed}"

command=(
  python -m optiq.train
  --config configs/mujoco/default.yaml
  --config "configs/mujoco/envs/${env_key}.yaml"
  --seed "$seed"
  --wandb-project "$project"
  --wandb-group "$group"
  --run-name "$run_name"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  command+=(--wandb-entity "$WANDB_ENTITY")
fi

printf 'task=%s env=%s seed=%s node=%s commit=%s\n' \
  "$task_id" "$env_name" "$seed" "$(hostname)" "$(git rev-parse HEAD)"
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${command[@]}"
