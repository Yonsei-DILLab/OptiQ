# Monge pilot: alternative GPU host

User authorized `heejoonorm@31.148.50.247:11717` on 2026-09-19 after login4
Slurm submission failed. Use this host for the existing 12-run Monge pilot.

- Numerical source remains `6510585036ab7221df0fc5022a970ef27e068061`, code ID
  `18d809bee424d0f94a3b6195a6f8d136e2b0a81647e46f16c3119c94e3dfce1f`.
- Base dependency remains `a08517ef9ed5fb8743252132997638b00feb5d75`, code ID
  `2c5908035f33763f536109bf34158f7413c34a69088eb3a30abf3f86ba91ecf8`.
- Both snapshots are copied from dildata and hash-verified. `OPTIQ_BASE` changes
  only the import path; no numerical source is edited. Runtime/launch source in
  this directory gets its own commit before GPU validation and execution.
- Python 3.11.13 in `/home/heejoonorm/.venvs/optiq-monge`; versions of JAX, CUDA
  wheels, NumPy, SciPy, Flax, Optax and other numerical packages match login4.
  Install torch 2.4.1+cpu separately from the official CPU wheel index (the
  experiment computes with JAX). Record the entire installed package list.
- Host has four RTX 3090 24GB GPUs, initially idle; use one process per GPU,
  with two distinct CPU cores per process, as in the login4 job allocation.
  Float32 parameters, highest matmul precision, same flags and source protocol.
- Run unchanged GPU validation first. Independent LP/assignment checks,
  original baseline parity, initial teacher equality, all-method finite updates,
  checkpoint continuation and diagnostic output must pass before main runs.
- Run the same 2 problems × 3 methods × 2 seeds × 35K updates. No result-based
  selection. Bounded runner stops after two hours with checkpoint signals if
  unfinished; it does not stop/destroy the paid instance. Do not retry failures
  silently. Record run IDs, process IDs, GPU, commit and exit status.
- Launch in the user's tmux session `optiq-monge`. This is a finite batch task;
  no web service, public port or system supervisor changes are needed.
- Preserve all other processes and the login4 queues. Only this pilot moves.

Host tree: `/home/heejoonorm/OptiQ/legacy_monge/{base,pilot,ops,runtime}`.
Central backup: `dildata:/data1/heejoonorm/OptiQ/studies/20260919_legacy_monge/remote_3114850247/`.
Use a dedicated restricted read-only rsync public key for dildata to pull this
tree; the private key stays on dildata and never enters the rented GPU host,
Git, logs, or reports. Keep virtual environment outside the backup tree.
Verify source manifests and completed run artifact hashes after synchronization.
No W&B credentials are needed for this diagnostic frozen-Q experiment.
