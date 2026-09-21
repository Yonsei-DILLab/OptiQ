#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /workspace/.venv-optiq-mujoco/bin/python "$repo_root/scripts/finish_v2_finite.py" \
  --protocol "$repo_root/outputs/v2_improvement/proximal_final_evaluation_protocol.json"
