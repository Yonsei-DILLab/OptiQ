#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
set -u
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
case "$1" in 0|1|2|3) seed="$1" ;; *) exit 2 ;; esac
export OPTIQ_CONFIG=mujoco_v2_finite CUDA_VISIBLE_DEVICES="$seed"
exec "$repo_root/scripts/run_v2.sh" "$seed" total_steps=1000000 \
    num_eval_episodes=10 diagnostic_interval=5000 checkpoint_interval=50000 \
    progress_bar=false "run_name=humanoid-v2-finite16-s${seed}"
