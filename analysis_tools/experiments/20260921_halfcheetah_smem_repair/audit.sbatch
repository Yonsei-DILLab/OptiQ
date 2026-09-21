#!/bin/bash
#SBATCH --job-name=hc-smem-audit
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_repair_20260921/logs/audit-%j.log
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 MUJOCO_GL=egl
unset PYTHONPATH LD_LIBRARY_PATH
exec /scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python -u audit.py \
  --campaign /scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921 \
  --output /scratch2/gsmin2024/research/optiq_halfcheetah_smem_repair_20260921/audit.json
