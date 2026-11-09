#!/usr/bin/env bash
# iBOLT + CrossQ critic + CrossQ actor training: best-of-K over stochastic z at the final (1M) checkpoint, 20 episodes.
cd ~/OptiQ
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=results/best_of_k/crossqAC
i=0
for r in outputs_tsweep/humanoid-ibolt-crossqAC-T0.5-s* outputs_pc/humanoid-ibolt-crossqAC-T0.5-s2_* outputs_tsweep/walker2d-ibolt-crossqAC-T0.1-s*; do
  CUDA_VISIBLE_DEVICES=$((i%4)) .venv/bin/python results/best_of_k/bok_eval_ibolt.py $r --out $O > /tmp/bok_$(basename $r | cut -d_ -f1).log 2>&1 & i=$((i+1))
done
wait; echo "BOK CROSSQAC DONE"
