#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$repo_root/scripts/run_gmm40.sh" \
  --batch-size 1 --num-policy-samples 256 --proposals-per-policy-sample 1 \
  --proposal-std 0.08 --temperature 0.25 --density-beta 0.1 \
  --group gmm40-optiq-256x256-std4-T0.25-beta0.1-30k \
  --output-root "$repo_root/outputs/gmm40_256_std4/runs" "$@"
