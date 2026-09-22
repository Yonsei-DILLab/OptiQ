#!/usr/bin/env bash
set -euo pipefail
: "${GMM40_PYTHON:?Set the isolated environment Python}"
: "${GMM40_OUTPUT:?Set the results directory outside Git}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for phase in preflight train; do
  pids=()
  for gpu in 0 1 2 3; do
    "$GMM40_PYTHON" -m experiments.gmm40_bandit.worker --output "$GMM40_OUTPUT" --gpu "$gpu" --phase "$phase" "$@" &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  if [ "$rc" != 0 ]; then exit "$rc"; fi
done
"$GMM40_PYTHON" -m experiments.gmm40_bandit.report --output "$GMM40_OUTPUT"
