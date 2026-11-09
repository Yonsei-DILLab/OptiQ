#!/usr/bin/env bash
# UTD=2 sweep (5090, 6 runs per GPU): iBOLT crossqAC (CrossQ critic 1024 + CrossQ-style actor, policy delay 3)
# T in {0.1, 0.25, 0.5} x {humanoid, walker2d, ant} x seeds {0,1}; SBX CrossQ (critic 1024) x 3 tasks x seeds {0,1}.
cd ~/OptiQ
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs outputs_utd2
i=0
for task in humanoid walker2d ant; do
  for s in 0 1; do
    CUDA_VISIBLE_DEVICES=$((i%4)) setsid -f ~/sbx-venv/bin/python baselines/crossq_sbx.py --task $task --seed $s --critic-width 1024 --utd 2 --output-root outputs_utd2 > logs/utd2-sbx-$task-s$s.log 2>&1 < /dev/null
    i=$((i+1))
    for T in 0.1 0.25 0.5; do
      CUDA_VISIBLE_DEVICES=$((i%4)) setsid -f .venv/bin/python train.py benchmark=$task +critic=crossq "alg.critic.hs=[1024,1024]" +actor=crossq temperature=$T seed=$s alg.utd=2 run_name=$task-ibolt-crossqAC-utd2-T$T-s$s output_root=outputs_utd2 > logs/utd2-ibolt-$task-T$T-s$s.log 2>&1 < /dev/null
      i=$((i+1))
    done
  done
done
echo launched $i
