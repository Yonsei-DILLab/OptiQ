#!/bin/bash
#SBATCH --job-name=kl-score-L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${STUDY_ROOT:?}" "${PARENT_ROOT:?}"
cd "$STUDY_ROOT/source"
unset LD_LIBRARY_PATH PYTHONPATH JAX_DEFAULT_MATMUL_PRECISION
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m experiments.kl_score_convergence.measure --root "$STUDY_ROOT" --parent "$PARENT_ROOT" --index "$SLURM_ARRAY_TASK_ID"
