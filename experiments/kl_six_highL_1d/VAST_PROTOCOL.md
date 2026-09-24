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
