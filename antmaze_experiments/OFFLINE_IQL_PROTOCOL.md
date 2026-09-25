# Uniform1M -> IQL -> frozen-critic extraction, seed0

One native AntMaze v1 simulator; original random XY training resets and500-step timeout.
Action i.i.d. Uniform([-1,1]^8) each step. Exactly1M transitions with terminal
next observations stored before reset; success masks0,time-limit masks1.
Reward100*nearest-goal Euclidean progress-1,success bonus0. No entropy reward,
NovelD,DACER or extra exploration. Prior live entropy experiments are untouched.
Dataset NPZ includes all actions/states/rewards/masks/episode-ends/success flags.
Record position coverage,goal successes and action moments. Never silently
substitute expert data or relabel unsuccessful random data as adequate coverage.

Official equation references:
https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py
https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py
https://github.com/ikostrikov/implicit_q_learning/blob/master/configs/antmaze_config.py
Expectile.9,V uses minimum target twinQ;then Q uses updatedV(nextstate).
gamma.99,targetEMA.005,Adam3e-4,batch256,250K offline Q/V updates.
Use existing scalar GELU256x2 twinQ and separate ReLU256x2 V.
This is an equation-level adaptation,not a verbatim official implementation.
No extra -1 preprocessing (already in actual reward),no reward normalization.

Freeze Q exactly,then100K actor-only updates of the current unmodified
Direct-GMM/TRG extraction,N=M64,beta1,T1,logstd[-5,-1],default mean aggregation.
IQL's AWR actor is not used. This is IQL critic + iBOLT extraction, not IQL policy.
Extraction state batches from the same fixed dataset,no rollout data added.
Every5K actor updates:100 fixed-origin full-pose/velocity rollouts with random z
per step,conditional noise disabled. Save coordinate NPZ,PNG and actor state.
Q/V saved after250K. Assert Q bytes unchanged during all actor optimization.
Q generalization to novel actor actions is NOT guaranteed by IQL's in-dataset
critic objective; random-data coverage and extrapolation remain limitations.

CPU collection starts immediately,then waits for an unlocked vast1 GPU without
stopping any existing worker/queue. Scratch real-data Q/V+actor preflight before
full optimization. Frozen source committed before execution. W&B offline
jaehun-antmaze/group uniform1M-IQL-frozenQ; sync on request,not a live uploader.
