#!/usr/bin/env bash
# iBOLT (CrossQ critic 1024 + CrossQ-style actor, policy delay 3), UTD 1, log-sigma range [-5, -3.5]
# (default is [-5, -1]; initial log-sigma set to the new upper bound -3.5, i.e. sigma <= 0.030).
# 5 MuJoCo tasks x seeds {0,1}, each task at the temperature used for its earlier crossqAC runs.
# Usage (on a server, repo root ~/OptiQ):  bash results/logstd/launch_logstd.sh [GPU list, default "0 1 2 3"]
#   runs are spread round-robin over the GPUs; outputs -> outputs_logstd/, logs -> logs/logstd-*.log
set -e
cd ~/OptiQ
GPUS=(${1:-0 1 2 3})
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs outputs_logstd
i=0
for tT in humanoid:0.5 walker2d:0.1 hopper:0.05 ant:0.25 halfcheetah:0.5; do
  task=${tT%:*}; T=${tT#*:}
  for s in 0 1; do
    CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]} setsid -f .venv/bin/python train.py benchmark=$task \
      +critic=crossq "alg.critic.hs=[1024,1024]" +actor=crossq temperature=$T seed=$s \
      alg.actor.custom_log_std_bounds=true alg.actor.log_std_min=-5.0 alg.actor.log_std_max=-3.5 alg.actor.initial_log_std=-3.5 \
      run_name=$task-ibolt-crossqAC-ls5to3.5-T$T-s$s output_root=outputs_logstd \
      > logs/logstd-$task-T$T-s$s.log 2>&1 < /dev/null
    i=$((i+1))
  done
done
echo "launched $i runs on GPUs ${GPUS[*]}"
