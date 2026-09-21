#!/bin/bash
#SBATCH --job-name=spline-energy-loss-pilot
#SBATCH --partition=base_suma_rtx3090,suma_rtx4090,gigabyte_a5000
#SBATCH --qos=base_qos
#SBATCH --exclude=node24,node48,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_spline_energy_bellman_20260922/logs/pilot-%j.out
#SBATCH --error=/scratch2/gsmin2024/research/optiq_spline_energy_bellman_20260922/logs/pilot-%j.err

set -euo pipefail
CAMPAIGN=/scratch2/gsmin2024/research/optiq_spline_energy_bellman_20260922
SOURCE="$CAMPAIGN/snapshot"
EXP="$SOURCE/analysis_tools/experiments/20260921_mujoco_spline_energy"
PYTHON=/scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_NAME WANDB_RUN_GROUP WANDB_RUN_ID WANDB_RESUME PYTHONPATH LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=offline WANDB_RUN_GROUP=spline_energy_relative_energy
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS=1
"$PYTHON" - "$CAMPAIGN" <<'PY'
import json, subprocess, sys
from pathlib import Path
p=Path(sys.argv[1]);s=p/'snapshot'
m=json.loads((p/'manifest.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=s,text=True).strip()==m['git_commit']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=s,text=True).strip()
assert json.loads((p/'checks/validation.json').read_text())['passed']
PY
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
cd "$EXP"
"$PYTHON" -u train.py --env Ant-v4 --seed 0 --total-steps 100000 \
  --loss-kind relative_energy --energy-scale 10 --energy-tail-start 4 \
  --output "$CAMPAIGN/outputs/ant-relative-energy-s0-100k" \
  --wandb-mode offline --require-gpu
