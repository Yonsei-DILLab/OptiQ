#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
set -u
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
case "$1" in
  0) gpu=0; fraction=0.25; label=proximal ;;
  1) gpu=2; fraction=0.0; label=fullstep ;;
  *) exit 2 ;;
esac
export OPTIQ_CONFIG=mujoco_v2_proximal CUDA_VISIBLE_DEVICES="$gpu"
exec "$repo_root/scripts/run_v2.sh" 0 total_steps=8000 \
  num_eval_episodes=2 eval_interval=1000 diagnostic_interval=500 checkpoint_interval=5000 \
  "alg.actor.soft_proximal_ess_fraction=$fraction" \
  output_root=outputs/v2_proximal_validation wandb.project=optiq_mujoco_v2_validation \
  wandb.job_type=validation progress_bar=false "run_name=humanoid-v2-${label}-gpu-smoke"
