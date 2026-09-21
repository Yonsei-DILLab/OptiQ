#!/bin/bash
#SBATCH --job-name=oq-moving-q
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --exclude=cs-gpu-01,node14,node23,node24,node31,node35,node40
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
ROOT=/lustre/hobbit9882/OptiQ-nonstationary-q-20260917
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/v5"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH
if [ "${1:-run}" = validate ]; then
    exec srun --cpu-bind=cores "$PY" -m experiment.validate --root "$ROOT" --size-index "$SLURM_ARRAY_TASK_ID"
fi
"$PY" - <<'CHECK'
import json
from pathlib import Path
r=Path('.');code=json.loads((r/'SOURCE_MANIFEST.json').read_text())['code_id']
for i in range(4):
    v=json.loads((r/f'VALIDATION_{i}.json').read_text())
    assert v['passed'] and v['source_code_id']==code
CHECK
srun --cpu-bind=cores "$PY" -m experiment.run --root "$ROOT" --index "$SLURM_ARRAY_TASK_ID" &
child=$!
stop_kind=none
trap 'stop_kind=timeslice; kill -USR1 "$child" 2>/dev/null || true' USR1
trap 'stop_kind=terminate; kill -USR1 "$child" 2>/dev/null || true' TERM INT
set +e
wait "$child"
code=$?
if [ "$stop_kind" != none ]; then
    wait "$child"
    code=$?
fi
if [ "$code" = 75 ] && [ "$stop_kind" = timeslice ]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$code"
