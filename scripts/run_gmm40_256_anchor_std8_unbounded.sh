#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$repo_root/scripts/run_gmm40_256_anchor_std8.sh" \
  --unbounded-actions \
  --group gmm40-optiq-256x1280-anchor-std8-unbounded-T0.25-beta0.1-30k \
  --output-root "$repo_root/outputs/gmm40_256_anchor_std8_unbounded/runs" "$@"
