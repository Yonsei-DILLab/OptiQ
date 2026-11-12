#!/usr/bin/env bash
set -euo pipefail
source_directory=$1
run_root=$2
python_executable=$3
cd "$source_directory"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export MPLCONFIGDIR="$run_root/mpl-cache"
mkdir -p "$run_root/runtime"
"$python_executable" - <<'PY' > "$run_root/runtime/packages.json"
import importlib.metadata, json, sys
print(json.dumps(dict(python=sys.version, packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}),indent=2))
PY
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv > "$run_root/runtime/hardware.csv"
"$python_executable" -m unittest adroit_online.test_protocol -v
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda "$python_executable" - <<'PY'
from adroit_online.controller import gpu_state
from adroit_online.run import source_provenance
source_provenance()
for g in (0,1):
    state=gpu_state()[g]
    assert not state['pids'], (g,'GPU occupied before preflight',state['pids'])
PY
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda "$python_executable" -m pytest -q \
    tests/test_config.py tests/test_distribution.py \
    -k 'not environment_update and not float32_support_stress' \
    > "$run_root/runtime/distribution-tests.log" 2>&1
"$python_executable" -m adroit_online.ibolt_preflight --root "$run_root/preflight"
exec "$python_executable" -m adroit_online.ibolt_controller --root "$run_root"
