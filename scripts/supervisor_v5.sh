#!/usr/bin/env bash
set -eo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -u
export OPTIQ_PYTHON="${OPTIQ_PYTHON:-/root/.venv-optiq-mujoco/bin/python}"
# Reuse the instance credential file; do not copy secrets into the v5 worktree.
export OPTIQ_ENV_FILE="${OPTIQ_ENV_FILE:-/root/OptiQ/.env}"
export OPTIQ_CONFIG="${OPTIQ_CONFIG:-mujoco_v5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${1:?Expected seed/GPU index 0..3}}"
cd "$repo_root"
exec /bin/bash scripts/run_v5.sh "$1" progress_bar=false
