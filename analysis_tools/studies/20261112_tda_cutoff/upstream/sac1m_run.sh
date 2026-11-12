#!/bin/bash
# Wait for SAC 1M checkpoints, then: extract mu/sigma on the iBOLT states -> exact table -> basin-graph persistence.
# usage: sac1m_run.sh <env> <gpu>   (env: hopper | humanoid). Writes marker ph_sac/SAC1M_<env>_DONE.
env=$1; g=$2
cd ~/OptiQ
CK=sac_1000000_steps
for ss in 0 1; do
  r=$(ls -d outputs_sac/$env-sac-s${ss}_*)
  while [ ! -f $r/checkpoints/$CK.zip ]; do sleep 120; done
done
sleep 30
cd ~/OptiQ
for ss in 0 1; do
  r=$(ls -d outputs_sac/$env-sac-s${ss}_*)
  for s in 0 1 2 3 4; do
    if [ $env = hopper ]; then sf=results/modal_em/data/hopper_s$s.npz; st=8; else sf=results/modal_em/data/humanoid_s${s}_n32768.npz; st=1; fi
    JAX_PLATFORMS=cpu ~/sbx-venv/bin/python results/modal_em/sac_ph_extract.py $r $CK results/modal_em/ph_sac/${env}_sac1m${ss}_on_s$s.npz $sf $st
  done
done
cd results/modal_em
../../.venv/bin/python sac_exact.py ph_sac/${env}_sac1m*_on_s*.npz > ph_sac/exact_${env}_1m.txt
M=20000; [ $env = humanoid ] && M=30000
for ss in 0 1; do for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$g ~/torch-venv/bin/python basin_ph.py sac ph_sac/${env}_sac1m${ss}_on_s$s.npz bp/sac1m${ss}_${env}_s$s.pkl --samples $M > logs_bp/sac1m${ss}_${env}_s$s.log 2>&1
done; done
../../.venv/bin/python bp_plot.py > /dev/null 2>&1
touch ph_sac/SAC1M_${env}_DONE
