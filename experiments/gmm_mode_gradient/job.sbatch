#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
set -euo pipefail
cd /scratch2/hobbit9882/OptiQ-ModeGradient-20260921
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 MPLBACKEND=Agg
unset LD_LIBRARY_PATH PYTHONPATH JAX_DEFAULT_MATMUL_PRECISION
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
exec srun --cpu-bind=cores "$PY" -m experiments.gmm_mode_gradient."$1"
