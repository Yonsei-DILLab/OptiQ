#!/usr/bin/env bash
set -euo pipefail
root="$1"
cd "$root/repo"
export CUDA_VISIBLE_DEVICES=3
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$root/repo"
exec /home/heejoonorm/.venvs/optiq-monge/bin/python -m experiments.gmm40_comparison.queue --root "$root"
