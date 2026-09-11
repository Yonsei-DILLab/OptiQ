#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260911_batch1_v1/summarize.py --campaign /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260911_batch1_v1 --require-complete
