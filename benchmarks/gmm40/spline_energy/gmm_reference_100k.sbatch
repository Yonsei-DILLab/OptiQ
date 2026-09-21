#!/bin/bash
#SBATCH --job-name=spline-gmm-ref
#SBATCH --array=0-4%5
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090,suma_a6000,suma_rtx4090,gigabyte_a6000,gigabyte_a5000,asus_6000ada,tyan_a6000,asus_a5000
#SBATCH --qos=base_qos
#SBATCH --exclude=node24,node48,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_spline_gmm_reference_20260922/logs/gmm-ref-%A_%a.out
#SBATCH --error=/scratch2/gsmin2024/research/optiq_spline_gmm_reference_20260922/logs/gmm-ref-%A_%a.err

set -euo pipefail
CAMPAIGN=/scratch2/gsmin2024/research/optiq_spline_gmm_reference_20260922
SOURCE="$CAMPAIGN/snapshot"
UPSTREAM=/scratch2/gsmin2024/research/optiq_latest_direct_comparison_20260921/upstream
TARGET=/scratch2/gsmin2024/research/optiq_latest_direct_comparison_20260921/target_data
PYTHON=/scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python
EXPECTED=$($PYTHON -c "import json;print(json.load(open('$CAMPAIGN/experiment_manifest.json'))['git_commit'])")
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$SOURCE" status --porcelain)"
$PYTHON -c "import json;assert json.load(open('$CAMPAIGN/checks/validation.json'))['passed']"
OUTPUT="$CAMPAIGN/outputs/seed${SLURM_ARRAY_TASK_ID}"
test ! -e "$OUTPUT"
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_NAME WANDB_RUN_GROUP WANDB_RUN_ID WANDB_RESUME PYTHONPATH LD_LIBRARY_PATH
export WANDB_MODE=offline PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
cd "$SOURCE"
"$PYTHON" -u benchmarks/gmm40/spline_energy/gmm_reference_experiment.py \
  --seed "$SLURM_ARRAY_TASK_ID" --steps 100000 --output "$OUTPUT" \
  --upstream "$UPSTREAM" --target-data "$TARGET" --wandb-mode offline --require-gpu
