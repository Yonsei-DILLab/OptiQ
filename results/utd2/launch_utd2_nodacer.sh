#!/usr/bin/env bash
# 5090: iBOLT crossqAC, UTD 2, DACER exploration off (dacer.enabled=false), seed 0. Humanoid T=0.5 (GPU0), Walker2d T=0.1 (GPU3).
cd ~/OptiQ
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs outputs_utd2
for tTg in humanoid:0.5:0 walker2d:0.1:3; do
  IFS=: read task T g <<< "$tTg"
  CUDA_VISIBLE_DEVICES=$g setsid -f .venv/bin/python train.py benchmark=$task +critic=crossq "alg.critic.hs=[1024,1024]" +actor=crossq temperature=$T seed=0 alg.utd=2 dacer.enabled=false run_name=$task-ibolt-crossqAC-utd2-nodacer-T$T-s0 output_root=outputs_utd2 > logs/utd2-nodacer-ibolt-$task-T$T-s0.log 2>&1 < /dev/null
done
