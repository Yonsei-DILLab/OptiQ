#!/bin/bash
#SBATCH --job-name=GMM3-means-min
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$STUDY_ROOT/source"
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
if [[ ${VALIDATION_ONLY:-0} == 1 ]]; then
 exec "$PY" -m experiments.gmm3_mean_local_minima.validate --out "$STUDY_ROOT/VALIDATION.json"
fi
exec "$PY" -m experiments.gmm3_mean_local_minima.run --root "$STUDY_ROOT" --index "$SLURM_ARRAY_TASK_ID"
