#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: launch_4way_100k.sh METHOD OUTPUT_DIRECTORY [TEMPERATURE]' >&2
  exit 2
fi
: "${OPTIQ_SOURCE_COMMIT:?set the immutable source commit}"
: "${OPTIQ_PYTHON:?set the validated Python interpreter}"
method="$1"
output_directory="$2"
temperature="${3:-3}"

exec "$OPTIQ_PYTHON" -m maze_benchmarks.run \
  --task 4way \
  --method "$method" \
  --output "$output_directory" \
  --seed 0 \
  --steps 100000 \
  --num-envs 16 \
  --updates-per-collect 16 \
  --batch-size 256 \
  --warmup 1024 \
  --eval-every 20000 \
  --eval-episodes 100 \
  --temperature "$temperature" \
  --source-commit "$OPTIQ_SOURCE_COMMIT"
