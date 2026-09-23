#!/bin/bash
#SBATCH --job-name=forward-wide
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
seed="$SLURM_ARRAY_TASK_ID"
cd "$STUDY_ROOT/source"
unset LD_LIBRARY_PATH PYTHONPATH JAX_DEFAULT_MATMUL_PRECISION
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$STUDY_ROOT/runtime/mpl_${SLURM_JOB_ID}"
/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m experiments.kl_forward_wide_1d.run --root "$STUDY_ROOT" --config "experiments/kl_forward_wide_1d/config.json" --index "$seed" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child" || true' USR1
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true' TERM INT
wait "$child"
