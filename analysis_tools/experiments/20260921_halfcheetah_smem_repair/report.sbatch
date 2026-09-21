#!/bin/bash
#SBATCH --job-name=hc-repair-report
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_repair_20260921/logs/report-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1
exec /scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python summarize.py \
  --campaign /scratch2/gsmin2024/research/optiq_halfcheetah_smem_repair_20260921 \
  --original /scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921 --plot
