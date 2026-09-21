#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 TASK [TASK ...]" >&2
  exit 2
fi

cd /workspace/DIME
for task in "$@"; do
  scripts/run_dog_adaptive_beta_ess16.sh "$task"
done
