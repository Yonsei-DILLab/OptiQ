#!/usr/bin/env bash
# Source this file from bash before running experiments.
_ibolt_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_ibolt_root/.venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export WANDB_MODE=offline
unset _ibolt_root
