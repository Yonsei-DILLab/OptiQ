#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
unset LD_LIBRARY_PATH
cd /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260911_batch1_v1/code
export PYTHONPATH="$PWD"
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m analysis_batch1.reference --campaign /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260911_batch1_v1
