#!/usr/bin/env bash
# Foreground worker; use the supplied supervisor template for long sweeps.
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
python_bin="${OPTIQ_PYTHON:-$(dirname "$repo_root")/.venv-optiq-mujoco/bin/python}"
[[ -x "$python_bin" ]] || { echo "Run scripts/setup_mujoco_env.sh first" >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
exec "$python_bin" -m scripts.mujoco_beta_sweep "$@"
