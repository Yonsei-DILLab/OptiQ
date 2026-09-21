#!/bin/bash
#SBATCH --job-name=optiq-mll-beta
#SBATCH --account=ys_ms
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/manfromearth_11/optiq-mll-beta-20260920/logs/%A_%a.out
#SBATCH --error=/scratch/manfromearth_11/optiq-mll-beta-20260920/logs/%A_%a.err
set -euo pipefail
campaign_root=/scratch/manfromearth_11/optiq-mll-beta-20260920
runtime=/scratch/manfromearth_11/optiq-resampled-cost50-20260920/venv
cd "$campaign_root/code/analysis_tools/experiments/20260920_mll_beta_anneal"
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD CUDA_HOME CUDA_PATH JAX_PLATFORMS JAX_PLATFORM_NAME XLA_FLAGS
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="/scratch/manfromearth_11/optiq-mujoco-anchor-r4r5-20260908/venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$runtime/bin:/usr/local/bin:/usr/bin:/bin"
export XDG_CACHE_HOME="$campaign_root/cache" WANDB_CACHE_DIR="$campaign_root/wandb-cache"
echo "Allocated node=$HOSTNAME job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ ${1:-} == smoke ]]; then
    exec "$runtime/bin/python" -u campaign.py smoke
else
    : "${SLURM_ARRAY_TASK_ID:?Explicit array index required}"
    exec "$runtime/bin/python" -u campaign.py "$SLURM_ARRAY_TASK_ID"
fi
