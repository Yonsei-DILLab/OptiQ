#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$repo_root/scripts/run_no_anchor.sh" \
  --config-name=optiq_dime_reach_proposal_anneal \
  --benchmarks reach_hard --seeds 0,2 "$@"
