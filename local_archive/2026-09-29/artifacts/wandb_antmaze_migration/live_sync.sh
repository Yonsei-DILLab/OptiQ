#!/usr/bin/env bash
set -euo pipefail
set -a
source /home/heechan/.env
set +a
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_PROJECT=antmaze WANDB_ENTITY=OptiQ WANDB_MODE=online
exec /home/heechan/.venv-ddiffpg-native/bin/python /home/heechan/OptiQ-ops/wandb-antmaze-migration-20260923/continue_moved_run.py "$@"
