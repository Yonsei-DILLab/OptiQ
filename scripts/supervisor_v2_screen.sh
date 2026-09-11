#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
set -u
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
index="$1"
export OPTIQ_CONFIG=archive/v2/original
case "$index" in
  0) name=optiq-reference; export OPTIQ_CONFIG=mujoco_setting
     overrides=(+alg.behavior_uniform_probability=0.0) ;;
  1) name=v2-otnll-T05; overrides=(alg.actor.distillation_loss=conditional_ot_nll alg.actor.temperature=0.5) ;;
  2) name=v2-otnll-T01; overrides=(alg.actor.distillation_loss=conditional_ot_nll alg.actor.temperature=0.1) ;;
  3) name=v2-mse-T01; overrides=(alg.actor.distillation_loss=pointwise_mse alg.actor.temperature=0.1) ;;
  *) exit 2 ;;
esac
exec "$repo_root/scripts/run_v2.sh" 0 "${overrides[@]}" \
  total_steps=100000 num_eval_episodes=3 diagnostic_interval=5000 checkpoint_interval=25000 \
  progress_bar=false "run_name=humanoid-screen-${name}-s0" \
  output_root=outputs/v2_screen wandb.project=optiq_mujoco_v2_screen \
  wandb.group=humanoid-v2-projection-temperature-screen wandb.job_type=screen
