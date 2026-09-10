#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${OPTIQ_PYTHON:-$(dirname "$repo_root")/.venv-optiq-mujoco/bin/python}"
seed="${1:---list}"
if [[ "$seed" == --list ]]; then
  for index in 0 1 2 3; do
    printf 'GPU %s: Humanoid-v4 seed %s; v2 semi-implicit actor; 16x64 OT; 1M steps\n' "$index" "$index"
  done
  exit 0
fi
case "$seed" in 0|1|2|3) ;; *) printf 'Usage: %s {0|1|2|3} [--check] [Hydra overrides]\n' "$0" >&2; exit 2 ;; esac
shift
export OPTIQ_ENV_FILE="${OPTIQ_ENV_FILE:-/workspace/OptiQ-heechan-no-anchor/.env}"
[[ -f "$OPTIQ_ENV_FILE" && -x "$python_bin" ]] || { printf 'Credential file or MuJoCo Python missing.\n' >&2; exit 1; }
export WANDB_ENTITY=OptiQ WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME WANDB_RUN_GROUP WANDB_NAME WANDB_PROJECT JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$seed}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { printf 'Allocate one GPU per worker.\n' >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo_root"
command=("$python_bin" run_optiq_dime.py "--config-name=${OPTIQ_CONFIG:-mujoco_v2}")
if [[ "${1:-}" == --check ]]; then
  shift
  command+=(--cfg job --resolve)
fi
exec "${command[@]}" benchmark=humanoid "seed=$seed" "$@"
