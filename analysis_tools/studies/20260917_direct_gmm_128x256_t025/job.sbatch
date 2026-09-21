#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=/lustre/hobbit9882/OptiQ-DirectGMM-128x256-T025-20260917
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export MUJOCO_GL=egl WANDB_ENTITY=OptiQ WANDB_MODE=online
unset LD_LIBRARY_PATH
if [ "$1" = validate ]; then
    exec srun --cpu-bind=cores "$PY" run.py validate
fi
exec srun --cpu-bind=cores "$PY" run.py run --env "$1" --seed "$SLURM_ARRAY_TASK_ID"
