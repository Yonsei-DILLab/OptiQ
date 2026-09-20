#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${OPTIQ_PYTHON:-python}"
seed="${1:-0}"
if [[ "$seed" == --help ]]; then
  printf 'Usage: OPTIQ_PYTHON=python bash scripts/run_v8.sh SEED [--check] [Hydra overrides]\n'
  exit 0
fi
[[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || { printf 'Seed must be a nonnegative integer.\n' >&2; exit 2; }
if [[ $# -gt 0 ]]; then shift; fi
check_only=false
if [[ "${1:-}" == --check ]]; then check_only=true; shift; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo_root"
# Configuration-only check never starts a tracking run or a GPU computation.
JAX_PLATFORMS=cpu "$python_bin" scripts/verify_v8.py "seed=$seed" "$@"
if [[ "$check_only" == true ]]; then exit 0; fi
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$python_bin" run_optiq_dime.py --config-name=mujoco_v8 "seed=$seed" "$@"
