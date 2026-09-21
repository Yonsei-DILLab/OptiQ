#!/bin/bash
set -euo pipefail
. "$(dirname "$0")/env.sh"
mkdir -p "$STUDY/runtime"
cd "$STUDY/pilot"
if [ "${1:-run}" = validate ]; then
    export CUDA_VISIBLE_DEVICES=0
    cpus=$("$PY" -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:2])))')
    exec taskset -c "$cpus" "$PY" validate.py
fi
exec "$PY" "$STUDY/ops/runner.py"
