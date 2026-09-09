#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${OPTIQ_PYTHON:-$(dirname "$repo_root")/.venv-optiq-no-anchor/bin/python}"
cd "$repo_root"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
exec "$python_bin" -m benchmarks.gmm40.train --require-gpu "$@"
