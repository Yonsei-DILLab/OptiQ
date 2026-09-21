#!/bin/bash
#SBATCH --job-name=spline-mj-smoke
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,tyan_a6000,asus_a5000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_mujoco_spline_energy_20260921/logs/smoke-%j.out
#SBATCH --error=/scratch2/gsmin2024/research/optiq_mujoco_spline_energy_20260921/logs/smoke-%j.err

set -euo pipefail
CAMPAIGN=/scratch2/gsmin2024/research/optiq_mujoco_spline_energy_20260921
SOURCE="$CAMPAIGN/snapshot"
EXP="$SOURCE/analysis_tools/experiments/20260921_mujoco_spline_energy"
PYTHON=/scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python
EXPECTED=$($PYTHON -c "import json;print(json.load(open('$CAMPAIGN/manifest/campaign.json'))['git_commit'])")
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$SOURCE" status --porcelain)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
cd "$EXP"
exec "$PYTHON" -u smoke.py --campaign "$CAMPAIGN"
