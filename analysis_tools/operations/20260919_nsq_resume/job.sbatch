#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --exclude=cs-gpu-01,node05,node14,node23,node24,node31,node35,node40
set -euo pipefail
ROOT="$1"
OPS="$2"
MODE="$3"
EVENT="$4"
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/v5:$ROOT/vendor"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export JAX_DEFAULT_MATMUL_PRECISION=highest
export POT_BACKEND_DISABLE_PYTORCH=1 POT_BACKEND_DISABLE_TENSORFLOW=1 POT_BACKEND_DISABLE_CUPY=1
unset LD_LIBRARY_PATH
if [ "$MODE" = validate ]; then
    dims=(1 2 4 8)
    exec srun --cpu-bind=cores "$PY" -m nsq.validate_legacy_td --root "$ROOT" --dim "${dims[$SLURM_ARRAY_TASK_ID]}"
fi
args=()
if [ "$MODE" = legacy ]; then args+=(--legacy-only); fi
"$PY" "$OPS/worker.py" --root "$ROOT" --event "$EVENT" "${args[@]}" &
child=$!
stop_kind=none
trap 'stop_kind=timeslice; kill -USR1 "$child" 2>/dev/null || true' USR1
trap 'stop_kind=terminate; kill -TERM "$child" 2>/dev/null || true' TERM INT
set +e
wait "$child"
code=$?
if [ "$stop_kind" != none ]; then wait "$child"; code=$?; fi
if [ "$stop_kind" = timeslice ]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$code"
