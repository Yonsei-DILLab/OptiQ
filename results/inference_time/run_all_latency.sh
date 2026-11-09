#!/usr/bin/env bash
# Measure every Table 2 method on one free RTX 5090.
# Usage (on the server): bash run_all_latency.sh <gpu index> [pid to pause while measuring]
# A training process sharing the machine can be paused (SIGSTOP) so the GPU is idle; it is
# always resumed (SIGCONT) on exit, also on failure. Pausing does not change its results.
GPU=$1; PAUSE=$2
cd ~/OptiQ/results/inference_time
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=2
if [ -n "$PAUSE" ]; then
  trap 'kill -CONT '"$PAUSE"'; echo "resumed $PAUSE"' EXIT
  kill -STOP "$PAUSE" && echo "paused $PAUSE"
  sleep 5
fi
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader -i $GPU
F='warn|deprecat|gym has been|upgrade to|migration guide'
echo "##### iBOLT / SAC / TD3 (jax, iBOLT venv)"
(cd ~/OptiQ && .venv/bin/python results/inference_time/bench_interleaved.py 2>&1 | grep -v -i -E "$F")
echo "##### SQL paper 2x256";        ~/tf-venv/bin/python    bench_baselines.py sql --hidden 256 2>&1 | grep -v -i -E "$F" | tail -n 3
echo "##### SQL official 2x128";     ~/tf-venv/bin/python    bench_baselines.py sql --hidden 128 2>&1 | grep -v -i -E "$F" | tail -n 3
echo "##### MEow";                   ~/torch-venv/bin/python bench_baselines.py meow 2>&1 | grep -v -i -E "$F" | tail -n 3
echo "##### DIPO";                   ~/torch-venv/bin/python bench_baselines.py dipo 2>&1 | grep -v -i -E "$F" | tail -n 4
echo "##### MFPO";                   ~/mfpo-venv/bin/python  bench_baselines.py mfpo 2>&1 | grep -v -i -E "$F" | tail -n 4
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader -i $GPU
echo "ALL LATENCY DONE"
