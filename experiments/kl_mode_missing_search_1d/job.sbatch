#!/bin/bash
#SBATCH --job-name=KL-mode-search
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
unset LD_LIBRARY_PATH PYTHONPATH JAX_DEFAULT_MATMUL_PRECISION
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$STUDY_ROOT/runtime/mpl_${SLURM_JOB_ID}"
cd "$STUDY_ROOT/source"
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
if [[ ${VALIDATION_ONLY:-0} == 1 ]]; then
  exec "$PY" -m experiments.kl_mode_missing_search_1d.validate --gpu --out "$STUDY_ROOT/runtime/GPU_VALIDATION.json"
fi
args=(--root "$STUDY_ROOT" --index "$SLURM_ARRAY_TASK_ID" --stage "${STAGE:-screen}")
if [[ -n ${SELECTED_CASE:-} ]]; then args+=(--selected-case "$SELECTED_CASE"); fi
exec "$PY" -m experiments.kl_mode_missing_search_1d.run "${args[@]}"
