#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
worker="${1:?GPU/seed index required (0, 1 or 2)}"
case "$worker" in
  0|1|2) ;;
  *) echo 'Worker must be 0, 1 or 2' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$worker"
exec "$repo_root/scripts/run_gmm40_native.sh" --seed "$worker"
