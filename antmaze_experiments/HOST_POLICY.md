# AntMaze server allocation

Latest user instruction, 2026-09-25, supersedes the earlier GMM40-only
reservation of `vast1` (RTX4090 x4). AntMaze work may use `vast1` and both
RTX5090 hosts, `vast-heechan-180` and `vast-heechan-199`, within the scope
of explicitly requested campaigns. For the current v3/v4 N=M128/256
experiment, use the placement and frozen source described in
`UTD256_NM_PROTOCOL.md`.

The previous 4090 Hopper GMM-TRG and NM GPU workers were stopped at the
user's instruction. Do not auto-resume them. Preserve their source, configs,
logs, checkpoints, W&B records, and NM/ablation data. Do not stop the server
instance or unrelated system services. Previous cancelled AntMaze services
remain cancelled. Existing T=3 v3/v4 learners on the 5090 hosts continue.

Read each host's `/etc/vast-agents-guide.md` before acting. Commit and push
exact experiment source before launch; use detached, clean frozen source
snapshots and respect per-GPU locks. The `vast1` canonical checkout has local
commits ahead of GitHub: do not reset or overwrite that branch while sharing
new source. Historical three-host manifests, archived supervisor configs,
and the 2026-09-25 GMM40 reservation describe past allocations only.

Later user correction, 2026-09-25: N=M256 is cancelled because its original
full-batch preflight exhausted GPU memory. Both later gradient-accumulation
N=M256 runs and sync services were also stopped, despite passing preflight.
Do not resume or retry them. N=M128 v3/v4 continue on vast1; T=3 v3/v4
continue on the two RTX5090 hosts. Keep all stopped N=M256 data and logs.
