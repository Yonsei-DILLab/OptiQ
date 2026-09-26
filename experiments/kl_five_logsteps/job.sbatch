#!/bin/bash
#SBATCH --job-name=KL5-early-1k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
unset LD_LIBRARY_PATH PYTHONPATH JAX_DEFAULT_MATMUL_PRECISION
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda
export JAX_THREEFRY_PARTITIONABLE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$STUDY_ROOT/source"
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m experiments.kl_five_logsteps.run \
 --root "$STUDY_ROOT" --parent "$PARENT_ROOT" --index "$SLURM_ARRAY_TASK_ID"
