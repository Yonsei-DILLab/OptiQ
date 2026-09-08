#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$repo_root/scripts/run_no_anchor.sh" \
  --config-name=optiq_scalar_mujoco_target_anneal \
  --benchmarks ant,half_cheetah --seeds 0 "$@"
