#!/bin/bash
# Wait for SAC seeds 2-4 1M checkpoints (per env), then extract on the iBOLT states and run basin-graph persistence
# with the matched setting (M=32768) into bp32k/. usage: sac5_post.sh <env> <gpu>
env=$1; g=$2; CK=sac_1000000_steps
cd ~/OptiQ
for ss in 2 3 4; do
  until r=$(ls -d outputs_sac_extra/$env-sac-s${ss}_* 2>/dev/null) && [ -f $r/checkpoints/$CK.zip ]; do sleep 120; done
done
sleep 30
for ss in 2 3 4; do
  r=$(ls -d outputs_sac_extra/$env-sac-s${ss}_*)
  for s in 0 1 2 3 4; do
    if [ $env = hopper ]; then sf=results/modal_em/data/hopper_s$s.npz; st=8; else sf=results/modal_em/data/humanoid_s${s}_n32768.npz; st=1; fi
    JAX_PLATFORMS=cpu ~/sbx-venv/bin/python results/modal_em/sac_ph_extract.py $r $CK results/modal_em/ph_sac/${env}_sac1m${ss}_on_s$s.npz $sf $st
  done
done
cd results/modal_em
../../.venv/bin/python sac_exact.py ph_sac/${env}_sac1m*_on_s*.npz > ph_sac/exact_${env}_1m_5seed.txt
for ss in 2 3 4; do for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$g ~/torch-venv/bin/python basin_ph.py sac ph_sac/${env}_sac1m${ss}_on_s$s.npz bp32k/sac1m${ss}_${env}_s$s.pkl --samples 32768 > logs_cmp/sac1m${ss}_${env}_s$s.log 2>&1
done; done
touch bp32k/SAC5_${env}_DONE
if [ -f bp32k/SAC5_hopper_DONE ] && [ -f bp32k/SAC5_humanoid_DONE ]; then ../../.venv/bin/python cmp5_table.py > /dev/null 2>&1; touch bp32k/ALL5_DONE; fi
