#!/usr/bin/env bash
# Paired fresh training seeds; candidate choice is explicit before launch.
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
set -u
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
variant="${OPTIQ_CONFIRMATION_VARIANT:?Select conditional64, unguarded256, or guarded256}"
case "$variant" in conditional64|unguarded256|guarded256) ;; *) exit 2 ;; esac
case "$1" in
  0) method=v2; seeds=(0 2) ;;
  1) method=optiq; seeds=(0 2) ;;
  2) method=v2; seeds=(1 3) ;;
  3) method=optiq; seeds=(1 3) ;;
  *) exit 2 ;;
esac
overrides=()
if [[ "$method" == optiq ]]; then
  export OPTIQ_CONFIG=mujoco_setting
  label=optiq-reference
else
  label="v2-$variant"
  overrides+=(alg.actor.temperature=0.1)
  if [[ "$variant" == conditional64 ]]; then
    export OPTIQ_CONFIG=mujoco_v2_conditional
  else
    export OPTIQ_CONFIG=mujoco_v2_guarded
    overrides+=(alg.actor.proposals_per_policy_sample=16)
    if [[ "$variant" == unguarded256 ]]; then
      overrides+=(alg.actor.soft_guard.enabled=false)
    fi
  fi
fi
for seed in "${seeds[@]}"; do
  # Any failure or SIGINT stops this worker before the next seed starts.
  "$repo_root/scripts/run_v2.sh" "$seed" "${overrides[@]}" \
    ++alg.behavior_uniform_probability=0.0 total_steps=1000000 \
    num_eval_episodes=10 diagnostic_interval=5000 checkpoint_interval=50000 \
    progress_bar=false output_root=outputs/v2_confirmation \
    wandb.project=optiq_mujoco_v2_confirmation wandb.job_type=confirmation \
    "wandb.group=Humanoid-v4_${label}_behavior0_4seed_1m" \
    "run_name=humanoid-${label}-s${seed}"
done
