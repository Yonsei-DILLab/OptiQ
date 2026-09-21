#!/bin/bash
#SBATCH --job-name=hc-smem-repair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_repair_20260921/logs/run-%j.log
set -euo pipefail
METHOD=$1
SEED=$2
STEPS=$3
PHASE=$4
EXPECTED=$5
GATE=${6:-false}
test "$(git rev-parse HEAD)" = "$EXPECTED"
test -z "$(git status --porcelain)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 MUJOCO_GL=egl
unset PYTHONPATH LD_LIBRARY_PATH WANDB_RUN_ID WANDB_RESUME WANDB_PROJECT WANDB_RUN_GROUP
IFS= read -r WANDB_API_KEY < /scratch2/gsmin2024/.secrets/optiQ_wandb
export WANDB_API_KEY WANDB_ENTITY=OptiQ WANDB_MODE=online
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
exec /scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python -u repair_train.py "$METHOD" \
  seed="$SEED" total_steps="$STEPS" require_gpu=true \
  learning_gate="$GATE" \
  comparison_phase="$PHASE" checkpoint_interval=10000 \
  run_name="hc-${METHOD}-${PHASE}-s${SEED}" \
  output_root=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_repair_20260921/outputs
