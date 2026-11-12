#!/bin/sh
set -eu
# source_dir data_file output_dir full_commit
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/heejoonorm/adroit-online-ibolt-20261112/venv/bin/python -u "$1/legacy_diagnose.py" --upstream "$1/upstream_basin_ph.py" --data "$2" --out "$3" --commit "$4"
