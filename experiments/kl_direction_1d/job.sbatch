#!/bin/bash
#SBATCH --job-name=kl1d-trg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
: "${STUDY_ROOT:?export STUDY_ROOT before submission}"
cd "$STUDY_ROOT/source"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR="$STUDY_ROOT/runtime/matplotlib_${SLURM_JOB_ID}"
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then
 exec "$PY" -m experiments.kl_direction_1d.validate --gpu --out "$STUDY_ROOT/runtime/GPU_VALIDATION.json"
fi
# Forward scheduler warning/termination; Python atomically saves full Adam/RNG.
"$PY" -m experiments.kl_direction_1d.run --root "$STUDY_ROOT" --index "$SLURM_ARRAY_TASK_ID" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child" || true' USR1
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true' TERM INT
wait "$child"
