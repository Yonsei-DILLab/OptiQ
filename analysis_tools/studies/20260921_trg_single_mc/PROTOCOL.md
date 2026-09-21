# direct-gmm-trg + SingleQ + MC64 (2026-09-21)

User-authorized new server: heejoonorm@103.177.249.208:37047, RTX5090 x4.
Base branch direct-gmm-trg, full SHA4ca69473515b5083d6e651845ded39da34e14538.
Actual branch implementation is analysis_tools/experiments/20260920_truncated_mll,
with shared framework/configs from analysis_tools/studies/20260918_nonstationary_nd/v5.
Do not run the legacy repository-root actor, or substitute old squashed Gaussian.

24 new runs: Humanoid-v4, HalfCheetah-v4, Ant-v4; each seed0–3; N64/M64 and N64/M256.
Ant seeds inferred0–3 consistently with the request. Each1M steps, default5K warmup.
Environment order is priority, not completion dependency; size conditions interleave
within each environment by seed. GPU4, up to2runs/GPU after full-size smoke checks,
4CPU/run affinity. Fresh run IDs; no automatic retry/resume of a failed run.

Actor unchanged from TRG branch: centers mu=tanh(raw network mean); conditional
diagonal Gaussian conditioned on [-1,1]^D; inverse-CDF action samples; no action
tanh transform. Log sigma[-5,-1], initial -1, sigma max0.367879. The teacher floor
exp(-5) is inactive over allowed student sigma, so actual conditional mixture
sampling and density correction match. N64 fresh normal latents; M IID candidates
from the uniform mixture (M256 is NOT exactly4 from each component). Teacher
w=softmax(Q_live/T-log q_F), T.25, beta1, stopped targets and weights. Direct
marginal action-space NLL includes differentiable truncation normalization.
No OT or Sinkhorn, no gradient clipping or external exploration noise.

Requested critic change only: one scalar Q and one Polyak target copy. Draw K64
independent full truncated policy actions at each next state, not bounded centers:
y=r+.99*(1-d)*mean_k Q_target(s',a'_k). Mean over actions, no critic min/entropy.
Actor teacher uses the single live Q after critic update. Batch256, UTD1, actor
delay1, Adam3e-4, target Polyak.005; architecture256x2 GELU. N,M,K are separate draws.
Dual center-only evaluation (zero-z/sampled-z) every5K with10episodes each; not
conditional truncated expectations. Branch defaults unchanged otherwise.
Checkpoint every50K, actor/critic/optimizer; no complete replay/RNG resume claim.

Freeze exact source, builder, configs, protocol, validators and queue on heejoon
before any training validation or benchmark. Record base/overlay SHA, per-file
hashes, resolved configs, PID/GPU/CPU allocation, W&B run URL in immutable sidecars.
Code-only smoke compilation/import checks before commit do not run optimization.
Any changed training configuration receives a new commit before execution.

Validation: K1/K64 analytic loss, actual optimizer, target/actor stop-gradient,
terminal mask, Polyak and routed path; TRG sampling/density/normalizer gradients;
all24 configs; six short GPU+W&B validation runs (3env x2sizes), actual batch256,
K64, total128steps/warmup32 (96updates), finite losses/metrics, correct critic_count
and backup_samples, sigma within branch limits, no OT calls. Smoke logs are a
separate validation group. Only then launch24 benchmarks and verify early online
steps. Do not wait until all benchmark runs finish.

RTX5090 requires an appropriate CUDA stack: isolated Python3.11/JAX0.6.2 CUDA12,
Flax0.10.4, Optax0.2.4, Gymnasium0.29.1/MuJoCo2.3.7/SB32.1.0 and CPU PyTorch2.4.1.
Record installed package lock and GPU device identity. No CUDA driver mutation.
Preserve branch numerical actor functions byte-for-byte; validate on real GPU.

W&B OptiQ/DirectGMM_heejoon, groups20260921_TRG_SingleQ_MC64_N64_M{64,256}_T025.
Keep credentials in0600 files outside source/logs/backups. dildata central storage:
/data1/heejoonorm/OptiQ/studies/20260921_trg_single_mc, periodically pull only the
campaign directory via a restricted read-only SSH key; no lab private key on Vast.
Disk16GB: monitor space before new jobs; pause pending queue below2GB free instead
of deleting checkpoints or silently altering scientific settings.
