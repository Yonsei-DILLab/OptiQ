#!/usr/bin/env bash
# Gamid-PG and AEAP latency on one free RTX 5090, same protocol as run_all_latency.sh, with the
# iBOLT/SAC/TD3 reference re-measured in the same session.
# Usage (on the server): bash run_gamid_aeap_latency.sh <gpu index> [pid to pause while measuring]
GPU=$1; PAUSE=$2
cd ~/OptiQ/results/inference_time
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
if [ -n "$PAUSE" ]; then
  trap 'kill -CONT '"$PAUSE"'; echo "resumed $PAUSE"' EXIT
  kill -STOP "$PAUSE" && echo "paused $PAUSE"; sleep 5
fi
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader -i $GPU
F='warn|deprecat|gym has been|upgrade to|migration guide'
echo "##### iBOLT / SAC / TD3 (reference, same session)"
(cd ~/OptiQ && .venv/bin/python results/inference_time/bench_interleaved.py 2>&1 | grep -v -i -E "$F")
echo "##### Gamid-PG";  ~/gamid-venv/bin/python bench_baselines.py gamid 2>&1 | grep -v -i -E "$F" | tail -n 4
echo "##### AEAP";      ~/torch-venv/bin/python bench_baselines.py aeap  2>&1 | grep -v -i -E "$F" | tail -n 5
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader -i $GPU
echo "GAMID AEAP LATENCY DONE"
