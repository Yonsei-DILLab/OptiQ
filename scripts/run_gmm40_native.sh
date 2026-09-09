#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$repo_root/scripts/run_gmm40.sh" \
  --batch-size 1 --num-policy-samples 256 --proposals-per-policy-sample 5 \
  --include-anchor --unbounded-actions --coordinate-scale 1 --proposal-std 8 \
  --temperature 0.25 --density-beta 0.1 \
  --group gmm40-optiq-native-256x1280-anchor-std8-T0.25-beta0.1-30k \
  --output-root "$repo_root/outputs/gmm40_native/runs" "$@"
