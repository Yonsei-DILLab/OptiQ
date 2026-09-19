#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --exclude=cs-gpu-01,node05,node14,node23,node24,node31,node35,node40
set -euo pipefail
ROOT="$1"
MODE="${2:-run}"
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export JAX_DEFAULT_MATMUL_PRECISION=highest
export POT_BACKEND_DISABLE_PYTORCH=1 POT_BACKEND_DISABLE_TENSORFLOW=1 POT_BACKEND_DISABLE_CUPY=1
unset LD_LIBRARY_PATH
cd "$ROOT"
if [ "$MODE" = validate ]; then
    exec "$PY" validate.py
fi
"$PY" -c 'import bootstrap,json; from pathlib import Path; p=json.loads(Path("validation/PASSED.json").read_text()); assert p["ok"] and p["source_code_id"]==bootstrap.verify()'
"$PY" experiment.py --index "$SLURM_ARRAY_TASK_ID" &
child=$!
stop_kind=none
trap 'stop_kind=timeslice; kill -USR1 "$child" 2>/dev/null || true' USR1
trap 'stop_kind=terminate; kill -TERM "$child" 2>/dev/null || true' TERM INT
set +e
wait "$child"
code=$?
if [ "$stop_kind" != none ]; then wait "$child"; code=$?; fi
exit "$code"
