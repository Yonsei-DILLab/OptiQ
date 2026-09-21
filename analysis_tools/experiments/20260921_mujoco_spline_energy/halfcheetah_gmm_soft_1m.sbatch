#!/bin/bash
#SBATCH --job-name=hc-gmm-soft-1m
#SBATCH --array=0-2%3
#SBATCH --partition=big_suma_rtx3090,suma_a6000,suma_rtx4090,gigabyte_a6000,gigabyte_a5000,asus_6000ada,tyan_a6000,asus_a5000
#SBATCH --qos=big_qos
#SBATCH --exclude=node24,node48,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_halfcheetah_gmm_soft_20260922/logs/train-%A_%a.out
#SBATCH --error=/scratch2/gsmin2024/research/optiq_halfcheetah_gmm_soft_20260922/logs/train-%A_%a.err

set -euo pipefail
CAMPAIGN=/scratch2/gsmin2024/research/optiq_halfcheetah_gmm_soft_20260922
SOURCE="$CAMPAIGN/snapshot"
PYTHON=/scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python
EXPECTED=$($PYTHON -c "import json;print(json.load(open('$CAMPAIGN/manifest.json'))['git_commit'])")
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$SOURCE" status --porcelain)"
for SEED in 0 1 2; do
  $PYTHON -c "import json;d=json.load(open('$CAMPAIGN/pilot/seed$SEED/COMPLETE.json'));assert d['completed'] and d['env_steps']==100000"
done
OUTPUT="$CAMPAIGN/outputs/seed${SLURM_ARRAY_TASK_ID}"
test ! -e "$OUTPUT"
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_NAME WANDB_RUN_GROUP WANDB_RUN_ID WANDB_RESUME PYTHONPATH LD_LIBRARY_PATH
export WANDB_MODE=offline WANDB_PROJECT=OptiQ-MuJoCo-Spline-Energy WANDB_RUN_GROUP=halfcheetah_gmm_soft
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
cd "$SOURCE/analysis_tools/experiments/20260921_mujoco_spline_energy"
"$PYTHON" -u train.py --profile gmm_soft --env HalfCheetah-v4 \
  --seed "$SLURM_ARRAY_TASK_ID" --output "$OUTPUT" --wandb-mode offline --require-gpu
