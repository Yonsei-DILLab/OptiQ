#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
set -u
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OPTIQ_CONFIG=mujoco_v2_guarded
case "$1" in
  0) enabled=true; seed=0 ;;
  1) enabled=false; seed=0 ;;
  2) enabled=true; seed=1 ;;
  3) enabled=false; seed=1 ;;
  *) exit 2 ;;
esac
exec "$repo_root/scripts/run_v2.sh" "$seed" \
  alg.actor.temperature=0.1 alg.actor.proposals_per_policy_sample=16 \
  "alg.actor.soft_guard.enabled=$enabled" \
  "run_name=humanoid-v2-K256-guard${enabled}-T0.1-s${seed}" \
  "wandb.group=Humanoid-v4_v2_K256_T0.1_guard${enabled}" \
  total_steps=100000 num_eval_episodes=3 diagnostic_interval=5000 checkpoint_interval=25000 \
  progress_bar=false
