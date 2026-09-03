#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 TASK [TASK ...]" >&2
  exit 2
fi

dependency_dir=/workspace/DIME/outputs/optiq_dime_dog_adaptive_beta_ess16_t0p25_sigma0p2_clip2p5/completed
required_tasks=(run trot walk stand)

while true; do
  ready=true
  for task in "${required_tasks[@]}"; do
    if [[ ! -f "$dependency_dir/${task}_seed1.done" ]]; then
      ready=false
      break
    fi
  done
  if [[ "$ready" == true ]]; then
    break
  fi
  sleep 60
done

export MINIMUM_SOURCE_ESS=32
exec /workspace/DIME/scripts/run_dog_adaptive_beta_queue.sh "$@"
