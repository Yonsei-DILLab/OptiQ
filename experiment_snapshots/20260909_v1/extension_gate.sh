#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
unset LD_LIBRARY_PATH
cd /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260909_v1/code
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m analysis_boltzmann.campaign --gate --campaign /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260909_v1 --stage extension
