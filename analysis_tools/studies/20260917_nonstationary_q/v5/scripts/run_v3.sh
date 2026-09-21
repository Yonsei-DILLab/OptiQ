#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${OPTIQ_PYTHON:-$(dirname "$repo_root")/.venv-optiq-mujoco/bin/python}"
seed="${1:---list}"
if [[ "$seed" == --list ]]; then
  for index in 0 1 2 3; do
    printf 'Seed %s: v3 Humanoid-v4; plain TD; no policy entropy/guard; teacher T=0.1 beta=1; 16x64 OT NLL; 1M steps\n' "$index"
  done
  exit 0
fi
case "$seed" in 0|1|2|3) ;; *) printf 'Usage: %s {0|1|2|3} [--check] [Hydra overrides]\n' "$0" >&2; exit 2 ;; esac
shift
check_only=false
if [[ "${1:-}" == --check ]]; then check_only=true; shift; fi
[[ -x "$python_bin" ]] || { printf 'Set OPTIQ_PYTHON to the pinned MuJoCo Python.\n' >&2; exit 1; }
if [[ -z "${OPTIQ_ENV_FILE:-}" && -f "$repo_root/.env" ]]; then
  export OPTIQ_ENV_FILE="$repo_root/.env"
fi
export OPTIQ_CONFIG="${OPTIQ_CONFIG:-mujoco_v3}"
case "$OPTIQ_CONFIG" in mujoco_v3|v3/final) ;; *) printf 'Use a v3 configuration.\n' >&2; exit 2 ;; esac
export WANDB_ENTITY="${WANDB_ENTITY:-OptiQ}" WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME WANDB_RUN_GROUP WANDB_NAME WANDB_PROJECT
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$check_only" == true ]]; then
  export JAX_PLATFORMS=cpu
else
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$seed}"
  [[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { printf 'Allocate one GPU per worker.\n' >&2; exit 1; }
fi
cd "$repo_root"
"$python_bin" scripts/verify_v3.py benchmark=humanoid "seed=$seed" "$@" >&2
command=("$python_bin" run_optiq_dime.py "--config-name=$OPTIQ_CONFIG")
if [[ "$check_only" == true ]]; then command+=(--cfg job --resolve); fi
exec "${command[@]}" benchmark=humanoid "seed=$seed" "$@"
