#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo 'Usage: launch_4way_t1_500k.sh OUTPUT_DIRECTORY' >&2
  exit 2
fi
: "${OPTIQ_SOURCE_COMMIT:?set the immutable source commit}"
: "${OPTIQ_PYTHON:?set the validated Python interpreter}"

exec "$OPTIQ_PYTHON" -m maze_benchmarks.run \
  --task 4way \
  --method optiq \
  --output "$1" \
  --seed 0 \
  --steps 500000 \
  --num-envs 16 \
  --updates-per-collect 16 \
  --batch-size 256 \
  --warmup 1024 \
  --eval-every 50000 \
  --eval-episodes 100 \
  --temperature 1 \
  --source-commit "$OPTIQ_SOURCE_COMMIT"
