#!/bin/bash
#SBATCH --job-name=hc-smemtr-smoke
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921/logs/smoke-%A_%a.log

set -euo pipefail
REPO=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921/source
CAMPAIGN=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921
EXP=$REPO/analysis_tools/experiments/20260921_halfcheetah_smem_tr
PYTHON=/scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python
EXPECTED=$($PYTHON -c "import json;print(json.load(open('$CAMPAIGN/manifest.json'))['git_commit'])")
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$REPO" status --porcelain)"
METHODS=(direct smem_tr)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
OUTPUT=$CAMPAIGN/smoke/$METHOD
test ! -e "$OUTPUT"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_NAME WANDB_RUN_GROUP WANDB_RUN_ID WANDB_RESUME PYTHONPATH LD_LIBRARY_PATH
export WANDB_MODE=disabled
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
cd "$EXP"
exec "$PYTHON" -u smoke.py "$METHOD" --output "$OUTPUT"
