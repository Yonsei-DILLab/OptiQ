#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${OPTIQ_PYTHON:-$(dirname "$repo_root")/.venv-optiq-mujoco/bin/python}"
seed="${1:---list}"
if [[ "$seed" == --list ]]; then
  for index in 0 1 2 3; do
    printf 'GPU %s: Humanoid-v4 seed %s; FINAL v2 checked; T=0.1 beta=1; 16x64 OT NLL + soft guard; 1M steps\n' "$index" "$index"
  done
  exit 0
fi
case "$seed" in 0|1|2|3) ;; *) printf 'Usage: %s {0|1|2|3} [--check] [Hydra overrides]\n' "$0" >&2; exit 2 ;; esac
shift
[[ -x "$python_bin" ]] || { printf 'MuJoCo Python missing; set OPTIQ_PYTHON or run scripts/setup_mujoco_env.sh.\n' >&2; exit 1; }
# Resolve credentials from this checkout, never an unrelated historical checkout.
if [[ -z "${OPTIQ_ENV_FILE:-}" ]]; then
  if [[ -f "$repo_root/.env" ]]; then
    export OPTIQ_ENV_FILE="$repo_root/.env"
  fi
fi
case "${OPTIQ_CONFIG:-mujoco_v2}" in
  mujoco_v2|mujoco_v2_checked|v2/final) ;;
  *) printf 'run_v2.sh is reserved for checked-K64. Use the direct runner with an explicit archive config for historical variants.\n' >&2; exit 2 ;;
esac
export WANDB_ENTITY="${WANDB_ENTITY:-OptiQ}" WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME WANDB_RUN_GROUP WANDB_NAME WANDB_PROJECT JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$seed}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { printf 'Allocate one GPU per worker.\n' >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo_root"
"$python_bin" scripts/verify_v2_final.py >&2
command=("$python_bin" run_optiq_dime.py "--config-name=${OPTIQ_CONFIG:-mujoco_v2}")
if [[ "${1:-}" == --check ]]; then
  shift
  command+=(--cfg job --resolve)
fi
exec "${command[@]}" benchmark=humanoid "seed=$seed" "$@"
