#!/usr/bin/env bash
set -euo pipefail
source_directory=$1
run_root=$2
shard=$3
python_executable="$run_root/venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
cd "$source_directory"
uv pip freeze --python "$python_executable" > "$run_root/runtime-lock.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv > "$run_root/hardware.csv"
"$python_executable" -m unittest adroit_online.test_protocol -v
"$python_executable" -m adroit_online.preflight --root "$run_root/preflight"
exec "$python_executable" -m adroit_online.controller --root "$run_root" --shard "$shard" --slots-per-gpu 2
