#!/usr/bin/env bash
# Finite foreground job for supervisor; retain its sibling log file.
set -euo pipefail
gpu="${1:?GPU index required}"
baseline="${2:?Baseline (meow or td3) required}"
seed="${3:?Random seed required}"
output="${4:?New output directory required}"
updates="${5:-100000}"
[[ "$gpu" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ && "$updates" =~ ^[0-9]+$ ]] || exit 2
[[ "$baseline" == "meow" || "$baseline" == "td3" ]] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate-experiments.sh
export CUDA_VISIBLE_DEVICES="$gpu"
# Training is PyTorch CUDA; JAX is imported only by shared evaluation helpers.
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$output")"
if [[ -e "$output" ]]; then
    printf 'Output already exists: %s\n' "$output" >&2
    exit 2
fi
python -u -m gmm40.baselines.run --baseline "$baseline" --seed "$seed" \
    --steps "$updates" --device cuda --output "$output" 2>&1 | tee -a "${output}.log"
