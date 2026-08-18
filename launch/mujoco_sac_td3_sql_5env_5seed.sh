#!/bin/bash
#SBATCH -J bStSQL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=5-00:00:00
#SBATCH --array=REQUIRED
#SBATCH -o slurm_logs/mujoco_sac_td3_sql_5env_5seed/%A_%a.out
#SBATCH -e slurm_logs/mujoco_sac_td3_sql_5env_5seed/%A_%a.err

set -euo pipefail

repo_root="/home/manfromearth_11/Online-OptiFlow/OptiQ"
parent_root="/home/manfromearth_11/Online-OptiFlow"
sql_root="$repo_root/.third_party/softqlearning-pytorch"
project="online_optiq_sac_td3_sql_h256x3_mujoco5_5seed"
mkdir -p "$repo_root/slurm_logs/mujoco_sac_td3_sql_5env_5seed"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate flowrl-ogbench
export WANDB_ENTITY="${WANDB_ENTITY:-online-optiflow}"
export WANDB_MODE="${WANDB_MODE:-online}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

task_id="${SLURM_ARRAY_TASK_ID}"
if [ "$task_id" -lt 0 ] || [ "$task_id" -ge 75 ]; then
  echo "Invalid task id: $task_id (expected 0-74)" >&2
  exit 1
fi

methods=(sac td3 sql)
env_names=(Hopper-v4 Walker2d-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4)
env_configs=(hopper walker2d halfcheetah ant humanoid)
env_steps=(1000000 1500000 3000000 3000000 5000000)
reward_scales=(30 10 30 300 100)

method_idx=$((task_id / 25))
within_method=$((task_id % 25))
env_idx=$((within_method / 5))
seed=$((within_method % 5 + 1))
method="${methods[$method_idx]}"
env_name="${env_names[$env_idx]}"
env_config="${env_configs[$env_idx]}"
total_steps="${env_steps[$env_idx]}"
reward_scale="${reward_scales[$env_idx]}"
group="${method}_${env_name}"
run_name="${method}_${env_name}_seed${seed}_h256x3_gelu_warmup10k_${total_steps}steps"

echo "task=$task_id method=$method env=$env_name seed=$seed steps=$total_steps"
echo "shared=h256x3_GELU_warmup10k_batch256_replay1m_lr3e-4_eval5k_10ep"
echo "project=$project group=$group run=$run_name node=$(hostname)"
if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$method" = "sac" ] || [ "$method" = "td3" ]; then
  cd "$parent_root"
  common_args=(
    --baseline_config "sbx_${method}"
    --env_config "$env_config"
    --seed "$seed"
    --total_steps "$total_steps"
    --learning_starts 10000
    --hidden_dim 256
    --hidden_layers 3
    --activation gelu
    --buffer_size 1000000
    --batch_size 256
    --gamma 0.99
    --tau 0.005
    --train_freq 1
    --gradient_steps 1
    --learning_rate 3e-4
    --eval_interval 5000
    --num_eval_episodes 10
    --wandb_project "$project"
    --wandb_entity "$WANDB_ENTITY"
    --wandb_mode "$WANDB_MODE"
    --wandb_group "$group"
    --run_name "$run_name"
    --device auto
  )
  if [ "$method" = "sac" ]; then
    python -m baselines.run_sbx "${common_args[@]}" \
      --action_noise none --ent_coef auto --target_entropy auto
  else
    python -m baselines.run_sbx "${common_args[@]}" \
      --action_noise normal --action_noise_std 0.1 \
      --target_policy_noise 0.2 --target_noise_clip 0.5 --policy_delay 2
  fi
else
  "$repo_root/scripts/setup_softqlearning_pytorch.sh" "$sql_root"
  cd "$sql_root"
  export WANDB_GROUP="$group"
  export WANDB_RUN_NAME="$run_name"
  python train.py \
    "config=$env_name" \
    "seed=$seed" \
    "steps=$total_steps" \
    warmup_steps=10000 \
    buffer_size=1000000 \
    batch_size=256 \
    hidden_sizes=256 \
    hidden_layers=3 \
    actor_lr=3e-4 \
    critic_lr=3e-4 \
    n_particles=16 \
    gamma=0.99 \
    tau=0.005 \
    "reward_scale=$reward_scale" \
    eval_every=5000 \
    test_num=10 \
    wandb=1 \
    "project=$project" \
    "description=$run_name"
fi
