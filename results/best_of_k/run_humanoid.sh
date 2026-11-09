#!/usr/bin/env bash
# Humanoid: iBOLT best-of-K (10 seeds, step 900K) and Gamid-PG/AEAP final checkpoints (5 seeds), 20 episodes each.
cd ~/OptiQ
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OMP_NUM_THREADS=2
O=results/best_of_k/humanoid
IB="outputs/humanoid-ibolt-T0.1-s0_20261107T145634Z_30a56111 outputs_relay/humanoid-ibolt-T0.1-s1_20261107T145712Z_5e3ee376
 outputs_relay/humanoid-ibolt-T0.1-s2_20261107T150059Z_9f0badc2 outputs_relay/humanoid-ibolt-T0.1-s3_20261107T150240Z_ed9a0f64
 outputs_relay/humanoid-ibolt-T0.1-s4_20261107T150939Z_4bdf4abf outputs_relay/humanoid-ibolt-T0.1-s5_20261107T123530Z_8945bda9
 outputs/humanoid-ibolt-T0.1-s6_20261107T123605Z_dd8efee1 outputs_relay/humanoid-ibolt-T0.1-s7_20261107T123543Z_610e11e5
 outputs/humanoid-ibolt-T0.1-s8_20261107T123617Z_e7a044ab outputs/humanoid-ibolt-T0.1-s9_20261107T123619Z_17f9f3e3"
i=0
for r in $IB; do CUDA_VISIBLE_DEVICES=$((i%4)) .venv/bin/python results/best_of_k/bok_eval_ibolt.py $r --out $O & i=$((i+1)); done
for r in outputs/humanoid-gamid-td3-n4-s*; do CUDA_VISIBLE_DEVICES=$((i%4)) ~/gamid-venv/bin/python results/best_of_k/bok_eval_baselines.py gamid $r --out $O & i=$((i+1)); done
for r in outputs_relay/humanoid-aeap-s*; do CUDA_VISIBLE_DEVICES=$((i%4)) ~/torch-venv/bin/python results/best_of_k/bok_eval_baselines.py aeap $r --out $O & i=$((i+1)); done
wait; echo "BOK HUMANOID DONE"
