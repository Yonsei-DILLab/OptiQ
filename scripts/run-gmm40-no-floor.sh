#!/usr/bin/env bash
# Run one finite job under supervisor; output and its sibling .log are retained.
set -euo pipefail
gpu="${1:?GPU index required}"
seed="${2:?Random seed required}"
output="${3:?New output directory required}"
[[ "$gpu" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate-experiments.sh
export CUDA_VISIBLE_DEVICES="$gpu"
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$output")"
if [[ -e "$output" ]]; then
    printf 'Output already exists: %s\n' "$output" >&2
    exit 2
fi
python -u -m gmm40.run --seed "$seed" --nm 256 --steps 100000 \
    --batch 256 --eval-samples 10000 --no-proposal-floor \
    --output "$output" 2>&1 | tee -a "${output}.log"
