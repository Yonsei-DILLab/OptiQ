#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
set -u
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
cd "$repo_root"
exec /workspace/.venv-optiq-mujoco/bin/python -u scripts/finish_v2_confirmation.py
