#!/bin/bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
STUDY=$(dirname "$REPO")
PY=/home/heejoonorm/.venvs/optiq-monge/bin/python
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$STUDY/jax_cache"
unset LD_LIBRARY_PATH
cd "$REPO"
if [ "${1:-}" = validate ]; then
    export CUDA_VISIBLE_DEVICES=0
    cpus=$("$PY" -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0))[:2])))')
    exec taskset -c "$cpus" "$PY" -m experiments.v1_heejoon_explorer.validate --output "$STUDY/validation"
elif [ "${1:-}" = start ]; then
    test -f "$STUDY/validation/VALIDATION_PASSED.json"
    test -f "$STUDY/WANDB_VERIFIED.json"
    test -f /home/heejoonorm/.config/optiq-secrets/wandb_api_key
    mkdir -p "$STUDY/runtime"
    suffix=$(basename "$STUDY")
    for environment in ant humanoid halfcheetah; do
        session="explorer-${environment}-${suffix:0:7}"
        if ! tmux has-session -t "$session" 2>/dev/null; then
            tmux new-session -d -s "$session" "$PY -m experiments.v1_heejoon_explorer.queue --env $environment --root '$STUDY' > '$STUDY/runtime/${environment}_worker.log' 2>&1"
        fi
    done
else
    echo 'Usage: launch.sh validate|start' >&2
    exit 2
fi
