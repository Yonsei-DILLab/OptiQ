#!/bin/bash
#SBATCH --job-name=KL-paper-score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
unset LD_LIBRARY_PATH PYTHONPATH JAX_DEFAULT_MATMUL_PRECISION
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$STUDY_ROOT/runtime/mpl_${SLURM_JOB_ID}"
cd "$STUDY_ROOT/source"
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m experiments.kl_paper_highL.measure --root "$STUDY_ROOT" --parent "$PARENT_ROOT" --index "$SLURM_ARRAY_TASK_ID"
