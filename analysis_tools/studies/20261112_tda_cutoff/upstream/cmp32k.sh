#!/bin/bash
# Matched comparison: iBOLT mixture N=32768 (both envs) vs SAC 1M, same states, same policy-sample count M=32768.
cd ~/OptiQ
mkdir -p results/modal_em/logs_cmp results/modal_em/bp32k
seq 0 8 511 > results/modal_em/data/hopper_idx_stride8.txt
for s in 0 1 2 3 4; do
  JAX_PLATFORMS=cpu .venv/bin/python results/modal_em/extract.py outputs_tsweep/hopper-ibolt-crossqAC-T0.05-s${s}_* \
    results/modal_em/data/hopper_s${s}_n32768.npz --n 32768 --states-from results/modal_em/data/hopper_s$s.npz \
    --idx results/modal_em/data/hopper_idx_stride8.txt > results/modal_em/logs_cmp/extract_hopper_s$s.log 2>&1 &
  EX="$EX $!"
done
cd results/modal_em
PY=~/torch-venv/bin/python; M=32768
(  # SAC first (fast), then Humanoid iBOLT
for e in hopper humanoid; do for ss in 0 1; do for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=3 $PY basin_ph.py sac ph_sac/${e}_sac1m${ss}_on_s$s.npz bp32k/sac1m${ss}_${e}_s$s.pkl --samples $M > logs_cmp/sac1m${ss}_${e}_s$s.log 2>&1
done; done; done
for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=3 $PY basin_ph.py ibolt data/humanoid_s${s}_n32768.npz bp32k/ibolt_humanoid_s$s.pkl --samples $M > logs_cmp/ibolt_humanoid_s$s.log 2>&1
done ) &
wait $EX  # extractions only
G=(3 2 0 3 2)
for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${G[$s]} $PY basin_ph.py ibolt data/hopper_s${s}_n32768.npz bp32k/ibolt_hopper_s$s.pkl --samples $M > logs_cmp/ibolt_hopper_s$s.log 2>&1 &
done
wait  # all
../../.venv/bin/python cmp_table.py > /dev/null 2>&1
touch bp32k/DONE
