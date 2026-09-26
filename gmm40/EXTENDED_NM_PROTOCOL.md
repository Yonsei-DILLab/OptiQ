# GMM40 extended N=M sweep, 2026-09-26

User replaces ongoing experiments with fixed-Q GMM40. Latest upstream
8a73d69; paper reference c429fbb. New seeds 0..3 for N=M in
1,4,16,64,256,1024,4096 (28 runs). Reuse verified prior N=M128 results.
All available Vast GPU models are authorized, superseding historical
4090-only constraints. Cancel the current PointMaze campaign, preserving data.

Unchanged: 100000 actor updates, batch256, 256x3 GELU, Adam3e-4, T1, beta1,
fresh 2D standard-normal z, mean-head variance16 fan-average uniform,
log sigma[-5,-3.5], initial-4, teacher-only sigma floor .05, no DACER.
Frozen Q=log original GMM40 density, a in [-1,1]^2 and x=40a; no critic,
replay, interaction or UTD. Original target/evaluation/reference hash retained.
Original evaluation schedule and 10000 full-policy plus mu-only samples remain.

Small shapes use byte-identical original learner. N>=1024 uses CloudTRG:
same full-batch random draws and candidate set, sequential per-cloud gradient
summation, division by256, then one Adam update. This changes floating-point
reduction order, not the mathematical loss or batch. Validate against dense
updates at small N and check actual-batch preflight/memory/time at each shape.
N=1 has a single teacher candidate so normalized importance weights are1.
Budgets are equal actor updates, not equal Q queries or wall time.

Dispatch: vast4 N1/4; vast3 N16/64; vast1 N256; vast2 N4096/1024.
Each host has four independently claiming GPU workers and no automatic retry.
Pending jobs are sorted seed-first (0, then 1, then 2, then 3). All sizes'
seed0 jobs are submitted first on their assigned hosts, without requiring a
long-running size to finish before other GPUs can start subsequent seeds.
Each job passes actual-shape preflight first, then starts fresh. Failed shape
jobs are held with logs; other sizes may continue. W&B OptiQ/gmm-trg with
distinct group gmm40-extended-nm-20260926. Preserve source SHA/config, proof,
checkpoints, both evaluation modes and optimizer-count audit.
