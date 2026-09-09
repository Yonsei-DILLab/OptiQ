#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
worker="${1:?GPU worker index required (0 or 1)}"
export CUDA_VISIBLE_DEVICES="$worker"
case "$worker" in
  0) seeds=(0 2) ;;
  1) seeds=(1) ;;
  *) echo 'Worker must be 0 or 1' >&2; exit 2 ;;
esac
for seed in "${seeds[@]}"; do
  "$repo_root/scripts/run_gmm40_256_anchor_std8_unbounded.sh" --seed "$seed"
done
