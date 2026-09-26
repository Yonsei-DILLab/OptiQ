#!/bin/bash
#SBATCH --job-name=KL-five-progress
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
unset LD_LIBRARY_PATH PYTHONPATH JAX_DEFAULT_MATMUL_PRECISION
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cuda JAX_THREEFRY_PARTITIONABLE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$STUDY_ROOT/runtime/mpl_${SLURM_JOB_ID}"
cd "$STUDY_ROOT/source"
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
if [[ ${VALIDATION_ONLY:-0} == 1 ]]; then
 exec "$PY" -m experiments.kl_five_progress.run --root "$STUDY_ROOT" --preflight "$POOL"
fi
exec "$PY" -m experiments.kl_five_progress.run --root "$STUDY_ROOT" --index "$SLURM_ARRAY_TASK_ID"
