# Humanoid crashed runs: fresh replacement runs

2026-09-18: user explicitly cancelled checkpoint continuation and requested only
the eight crashed Humanoid runs be run again from initialization.

- Direct GMM N128, M256; temperatures 0.5 and 0.25; seeds 0,1,2,3.
- 1,000,000 environment steps, uniform warmup 5,000, original batch256,
  optimizer, architecture, evaluation (every5K, two modes ×10 episodes), and
  model checkpoint interval50K are unchanged. No checkpoint is loaded.
- Reuse the verified frozen numerical source and resolved configuration from
  each original campaign; CPU preflight checks equality except output/name and
  provenance. Existing GPU validation is reused because numerical paths and
  learning settings are unchanged.
- Original source ID:
  `fd918514023ae5ce20a91db53f19801549fd9d249e46e963b643d498834f5f28`.
- W&B: `OptiQ/DirectGMM_heejoon`, original temperature-specific group, fresh
  run ID and `fresh2` in run name. Config records the replaced W&B ID. Old
  crashed runs, logs, and checkpoints are preserved.
- Results: original campaign `outputs/humanoid_fresh_20260918_s{seed}`;
  status: `status/humanoid_fresh_20260918_s{seed}.json`.
  Existing dildata collectors therefore copy and hash-check completed results.
- Commit this operational bundle on `heejoon` before launching. Record both
  new launcher commit and original source/campaign commit, submitted IDs and
  config/source hashes. Do not mutate the frozen source or original wrappers.
- Two independent arrays, each seed0–3, GPU1/CPU2/RAM24GB per task,24h limit,
  no completion dependencies, no automatic requeue. Use available `base_qos`
  capacity (up to8 GPUs) on supported ordinary GPU partitions. This higher
  priority QoS does not guarantee immunity from all scheduler/node failures.
- Failed Ant runs remain deferred. Other running experiments are untouched.

Campaign roots:

| temperature | login4 root | dildata study |
|---|---|---|
|0.5|`/lustre/hobbit9882/OptiQ-DirectGMM-128x256-20260917`|`/data1/heejoonorm/OptiQ/studies/20260917_direct_gmm_128x256`|
|0.25|`/lustre/hobbit9882/OptiQ-DirectGMM-128x256-T025-20260917`|`/data1/heejoonorm/OptiQ/studies/20260917_direct_gmm_128x256_t025`|
