#!/usr/bin/env bash
set -euo pipefail
root="$1"
environment="$2"
cd "$root/repo"
export PYTHONPATH="$root/repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/heejoonorm/.venvs/optiq-monge/bin/python -m experiments.v2_heejoon_explorer.queue --env "$environment" --root "$root"
