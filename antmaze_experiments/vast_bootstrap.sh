#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/antmaze-temperature-20260924
SOURCE="$ROOT/code"
BASE=${ANTMAZE_BASE_PYTHON:?}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export DEBIAN_FRONTEND=noninteractive
if [ ! -x "$ROOT/venv/bin/python" ]; then
  "$BASE" -m venv --system-site-packages "$ROOT/venv"
fi
if [ ! -f "$ROOT/runtime-ready.json" ]; then
  apt-get update -qq
  apt-get install -y -qq libosmesa6-dev libgl1-mesa-dev patchelf gcc g++
  uv pip install --python "$ROOT/venv/bin/python" --no-cache 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$ROOT/venv/bin/python" --no-cache 'numpy==1.26.4' 'gym==0.23.1' 'Cython==0.29.37' 'mujoco-py==2.1.2.14' 'h5py' 'dm-control' 'pybullet' 'termcolor'
  uv pip install --python "$ROOT/venv/bin/python" --no-deps --no-cache 'd4rl==1.1'
  if [ ! -f "$ROOT/mujoco210/bin/libmujoco210.so" ]; then
    curl -fL --retry 3 https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -o "$ROOT/mujoco210.tar.gz"
    tar -xzf "$ROOT/mujoco210.tar.gz" -C "$ROOT"
  fi
  export MUJOCO_PY_MUJOCO_PATH="$ROOT/mujoco210"
  export LD_LIBRARY_PATH="$ROOT/mujoco210/bin:${LD_LIBRARY_PATH:-}"
  export MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1
  cd "$SOURCE"
  "$ROOT/venv/bin/python" -m antmaze_experiments.vast_runtime_check
fi
cd "$SOURCE"
exec "$ROOT/venv/bin/python" -u -m antmaze_experiments.vast_queue run
