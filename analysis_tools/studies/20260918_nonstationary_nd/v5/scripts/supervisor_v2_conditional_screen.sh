#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
set -u
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OPTIQ_CONFIG=mujoco_v2_conditional
case "$1" in
  0) temperature=0.1; seed=0 ;;
  1) temperature=0.5; seed=0 ;;
  2) temperature=0.1; seed=1 ;;
  3) temperature=0.5; seed=1 ;;
  *) exit 2 ;;
esac
exec "$repo_root/scripts/run_v2.sh" "$seed" "alg.actor.temperature=$temperature" \
  total_steps=100000 num_eval_episodes=3 diagnostic_interval=5000 checkpoint_interval=25000 \
  progress_bar=false
