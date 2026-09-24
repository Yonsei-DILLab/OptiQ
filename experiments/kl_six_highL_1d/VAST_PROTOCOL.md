# Vast RTX5090 execution extension, 2026-09-25

The user authorized host heejoonorm@38.49.42.46:60616 and task-relevant data/key
transfers. The four GPUs were idle. Move a disjoint subset of unstarted indices
from Slurm2324996 only after a successful GPU preflight. Record exact ownership
in VAST_ALLOCATION.json, cancel only the corresponding PENDING elements, and
verify cancellation before starting Vast workers. Existing running jobs remain.

Numerical source/config remains the six-target confirmation from
41a020577dbf8a87d8f96d714ab3888a4bb82260. This extension makes the existing Forward
parent path configurable and adds a locked one-process-per-GPU worker and backup.
No numerical source, loss, L, N, M, batch, optimizer, target or sample count changes.
Existing Forward checkpoint hash/provenance and initial actor parameters must match.

Hardware runtime: existing Vast JAX0.6.2/flax0.10.4/optax0.2.4 versus login4
JAX0.4.33/flax0.9.0/optax0.1.7. Set JAX_THREEFRY_PARTITIONABLE=false explicitly to
match the older JAX random generator default; otherwise initial actors differ.
Validation requires all four original parameter hashes and checkpoint restoration,
small-L gradient identity of the inherited adapters, true L1048576 training and
1M-action evaluation. Floating point training need not be bitwise identical across
GPU/compiler versions; record host/runtime in results. Persistent compile caching
only avoids repeated compilation. Evaluation functions and criteria are unchanged.

Use tmux workers with independent index lists; one GPU per run. Each worker stops
on an unexpected failure, preserving the remaining jobs and latest full-state
checkpoint for inspection. The existing training loop resumes unchanged checkpoints.
Store source manifest and deployment commit before any new training.

Results live inside the already-authorized read-only backup root:
/home/heejoonorm/OptiQ/gmm40_bandit/results/kl_six_highL_20260925.
Dildata pulls only this subfolder into the existing study's vast_campaign every
180 seconds, excluding transient compile cache. No credentials need copying and
no new authorized_keys entry is required. Keep login4 campaign and Vast campaign
separate so immutable provenance and run ownership remain unambiguous.

The optional deployment xla_flags field is recorded explicitly. A preflight may
use --xla_gpu_enable_triton_gemm=false to avoid very slow Blackwell Triton GEMM
autotuning, selecting library GEMM without changing the mathematical update or
precision requirements. Only a validated flag combination is used for training.

Preflight progress and a Python traceback every90s diagnose compiler stalls.
Steady-state timing reuses the SAME5-update compiled block rather than timing a
new10-update shape. If needed, --xla_disable_hlo_passes=constant_folding disables
a compile-time optimization, not any part of the runtime mathematical update;
validate and record the complete XLA_FLAGS before training.

## Selected validated runtime
JAX/jaxlib/CUDA plugin/PJRT 0.5.3 in `/home/heejoonorm/OptiQ/env-kl-jax053`;
Flax0.10.4, Optax0.2.4, NumPy2.2.6, SciPy1.15.3 supplied by the pre-existing
GMM40 JAX environment through a .pth entry. Both environment paths must be
preserved. Initialization hashes for seeds0–3 match the original exactly.
Training flags: `JAX_THREEFRY_PARTITIONABLE=false` and
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_NO_UNROLL`.
Do not use the stalled JAX0.6.2 preflight for training. On RTX5090 the selected
runtime passed both oracle types at the actual batch32/N128/M128/L1048576,
1M-action evaluation, all12 small-L gradient parity checks and resume/RNG checks.
Warmed5-update blocks measured0.234–0.236seconds/update. GPU/compiler runtime
is recorded explicitly; bitwise cross-hardware training equality is not claimed.

Migration: retain Slurm2324996 indices0–7 onlogin4, move only PENDING indices8–23
to Vast. Four GPU workers own `[8,12,16,20]`, `[9,13,17,21]`, `[10,14,18,22]`,
`[11,15,19,23]`, respectively. This places all four seeds of a case together.
Commit this allocation protocol before migration and training. Keep both source
commits in the central deployment record. Cancel moved pending jobs before workers
start and record cancellation evidence; never cancel the eight running jobs.
