# NovelD 0.1 on v2/v3/v4

User authorization 2026-09-22: v2, v3, v4, each with OptiQ/SAC/DIPO/MFPO,
NovelD coefficient 0.1, training seed0, 1M environment interactions. Twelve
fresh runs. Preserve the separately authorized v1 DIPO coefficient0.01 run and
all previous results; do not relaunch cancelled campaigns.

All existing dense environment/RND/NovelD/policy/optimizer settings remain.
No sparse reward conversion, DDiffPG mode-specific Q, ERIR, xy-only RND, reward
normalization or directional shaping. The reference to DDiffPG concerns its
NovelD convention; this is not a full DDiffPG replication.
Batch256, UTD1, replay1M, one environment. Native model/LR settings retained.
OptiQ: 256x2 GELU, T.25, beta1, DACER on, mean-init1, random latent, N=M64,
log sigma[-5,-1]/initial-1; other methods retain their native settings.
Evaluate every250k; full state checkpoint only final1M. Final100 direct-policy
and native rollouts for each natural/fixed-full-state condition. No additional
DACER noise or intrinsic reward at evaluation. Individual training seed0 policies
must not be pooled to claim multimodal behavior.

Two six-job shards across servers180/199, four GPU slots each. Existing GPU locks
are respected, including v1 DIPO on180. Slow DIPO jobs start early. Each job's
272-step smoke and new-process280-step restore verification gates only its own
production run. There is no all-method, all-preflight, or all-maze barrier.
Every completed job releases its slot and the next eligible queued job starts
within the controller's2-second polling interval, regardless of maze/method.
Normal completion/backfill is automatic. Any job failure holds that shard's
pending jobs, preserves live workers, and records the error; no automatic restart.

Record exact frozen training commit, submodule hashes, actual configs, complete
resume state and raw trajectories. On completion collect and verify local archives,
then compare spatial coverage, exploration depth, occupancy concentration and
direct-policy route distributions; success and return are secondary metrics.
Keep training exploration and learned-policy route retention separate.
