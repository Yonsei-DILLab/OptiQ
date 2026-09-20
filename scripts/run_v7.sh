#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${OPTIQ_PYTHON:-$(dirname "$repo_root")/.venv-optiq-mujoco/bin/python}"
seed="${1:---list}"
if [[ "$seed" == --list ]]; then
  printf 'v7: 4096 Gaussian integration points; 256 fresh teacher latents, one draw each; teacher 256->16 importance resampling; persistent state-conditioned dual MLP, one Adam step at 1e-4; 4096x16 assignments; 16 joint-sampled actor queries; conditional SAC actor; T=.25; SAC soft TD with fixed alpha=T; unchanged v5 actor/critic LR/UTD/frequency; 1M steps\n'
  exit 0
fi
[[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || {
  printf 'Usage: bash %s SEED [--check] [Hydra overrides]\n' "$0" >&2
  exit 2
}
shift
check_only=false
if [[ "${1:-}" == --check ]]; then check_only=true; shift; fi
[[ -x "$python_bin" ]] || { printf 'Set OPTIQ_PYTHON to the pinned MuJoCo Python.\n' >&2; exit 1; }
export OPTIQ_CONFIG="${OPTIQ_CONFIG:-mujoco_v7}"
export WANDB_ENTITY="${WANDB_ENTITY:-OptiQ}" WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME WANDB_RUN_GROUP WANDB_NAME WANDB_PROJECT
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$check_only" == true ]]; then
  export JAX_PLATFORMS=cpu
else
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$((seed % 4))}"
  [[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { printf 'Allocate one GPU per worker.\n' >&2; exit 1; }
fi
cd "$repo_root"
"$python_bin" scripts/verify_v7.py "seed=$seed" "$@" >&2
command=("$python_bin" run_optiq_dime.py "--config-name=$OPTIQ_CONFIG")
if [[ "$check_only" == true ]]; then command+=(--cfg job --resolve); fi
exec "${command[@]}" "seed=$seed" "$@"
