#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -u
export OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python
export OPTIQ_ENV_FILE=/root/OptiQ/.env
export OPTIQ_CONFIG=mujoco_v2 WANDB_ENTITY=OptiQ
export CUDA_VISIBLE_DEVICES="${1:?Expected seed/GPU index 0..3}"
cd /root/OptiQ
exec /bin/bash scripts/run_v2.sh "$1" progress_bar=false
