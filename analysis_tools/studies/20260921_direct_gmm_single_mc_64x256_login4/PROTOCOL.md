# DirectGMM SingleQ · N64 M256 K64 · Humanoid · login4

User request 2026-09-21: temperature0.25, N=K=64, M=256, Humanoid seed0–4 on login4.
Five fresh 1M-step runs. Existing Vast experiments and deferred login4 jobs remain untouched.

## Algorithm

Use the same learned-sigma Direct GMM actor and reward-only single-critic update as
heejoon commit `9ac88463efc44cf2a084458c322f6f6ae130d6f9`.

\[
a'_{bk}\sim\pi_\theta(\cdot\mid s'_b),\quad
y_b=\operatorname{sg}\left[r_b+0.99(1-d_b)\frac1{64}
\sum_{k=1}^{64}Q_{\bar\phi}(s'_b,a'_{bk})\right],\qquad
L_Q=\frac1B\sum_b(Q_\phi(s_b,a_b)-y_b)^2.
\]

N=64 fresh latent-conditioned Gaussians form the actor mixture. M=256 teacher
candidates are sampled IID from its conditional-mixture proposal, using
`proposals_per_policy_sample=4`. This is a total sample-count multiplier, not
stratified four-per-component sampling. Teacher-only sigma floor0.05 and matching
squashed mixture density correction remain unchanged.

\[
w_j=\operatorname{softmax}_j\left(Q_\phi(s,b_j)/0.25-\log q_F(b_j\mid s)\right),\qquad
L_\pi=-\sum_{j=1}^{256}w_j\log\left[\frac1{64}\sum_{i=1}^{64}k_\theta(b_j\mid s,z_i)\right].
\]

Teacher weights and candidates are detached. The TD64 actions use the actual full
stochastic policy, separately from the teacher candidates; no proposal floor,
twin-min, importance weights, or entropy reward enter TD. Critic then actor are
updated once per environment step. The existing actor/critic numerical functions
are byte-identical to the baseline; only the entry-point N=M=K validation restriction
is relaxed to allow positive integer teacher multipliers, and the config changes.

| Setting | Value |
|---|---|
| Environment / seeds | Humanoid-v4 / 0,1,2,3,4 |
| Steps / uniform warmup | 1M / 5K |
| N / M / K / temperature | 64 / 256 / 64 / 0.25 |
| Actor / critic architecture | 256×2 GELU / single scalar 256×2 GELU |
| Actor sigma | Learned; initial0.5; log sigma[-5,1] |
| Replay / actor batch | 256 / 256 |
| Optimizer | Adam3e-4; no gradient clipping |
| Gamma / target Polyak | 0.99 / 0.005 |
| Evaluation | Original dual mean-only, every5K,10episodes each |
| Checkpoint | Every50K; actor/critic/optimizer; no full replay/RNG resume |
| W&B | OptiQ/DirectGMM_heejoon |
| W&B group | 20260921_DirectGMM_SingleQ_MC64_N64_M256_T025_login4 |

## Validation and scheduling

Commit source builder, configuration, launch scripts, protocol before GPU training.
Verify base hashes and prove actor/critic numerical files unchanged. Resolve all5
Hydra configs and check existing login4 W&B authentication. Run a128-step Humanoid
GPU validation with32-step warmup at actual batch256/N64/M256/K64; require96updates,
finite losses and MC statistics, correct critic_count1/backup_samples64/entropy0,
and online W&B completion. Validation uses a separate group, not a benchmark seed.
One common validation gate is followed by a five-seed array: seeds have no mutual
completion dependency and are concurrently eligible. Each seed requests GPU1,
CPU2, RAM24GB,24h, no requeue. Use non-preempting big_qos and healthy compatible
partitions; scheduler fit may limit immediate concurrency. Preserve previously
held/interrupted jobs. Record full source commit, manifest, commands, node, Slurm
IDs and W&B IDs. Do not silently retry partial training or substitute seeds.

## Storage

Initial deployment to Lustre failed before registration or training: the user's1TB
Lustre soft quota is exceeded (1.001TB, grace expired). No old experiments or
checkpoints were deleted. Use the shared scratch2 filesystem instead, whose actual
writeability and user quota are checked before deployment.

Login4 root: `/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921`.
Central source/run data and index: `dildata:/data1/heejoonorm/OptiQ/studies/20260921_direct_gmm_single_mc_64x256_login4`.
A dedicated dildata collector pulls this folder every3minutes using a new
campaign-scoped read-only rrsync key. The private key stays on dildata; only its
public key is installed on login4 with forced rrsync -ro, no forwarding or PTY.
Existing backup keys and collectors remain unchanged. Completed artifact hashes
are verified centrally. Keys, checkpoints, logs and generated records stay outside
Git. Runtime metadata records the new full commit; the original SingleQ source
commit remains an explicit build input. The failed Lustre deployment is not used.

Hardware health probe2302293 reconfirmed that cs-gpu-01 cannot expose a usable CUDA
device despite advertising5free GPUs. Keep it excluded along with the previously
identified incompatible nodes. Do not launch CPU fallback training.
