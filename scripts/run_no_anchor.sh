#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${OPTIQ_PYTHON:-$(dirname "$repo_root")/.venv-optiq-no-anchor/bin/python}"
[[ -x "$python_bin" ]] || { echo "Run scripts/setup_no_anchor_env.sh first" >&2; exit 1; }
cd "$repo_root"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
exec "$python_bin" -m scripts.no_anchor_runs "$@"
