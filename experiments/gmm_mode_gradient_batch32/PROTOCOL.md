# 3-mode responsibility gradient: batch 32 repeat

User request: repeat the latest responsibility-routing experiment with batch 32.
This is a separate campaign. Do not overwrite the batch-1 runs or running source.

## Comparison and fixed settings

- Methods: original Direct GMM (`baseline`), winning-mode gradient only (`mode_only`), winning-mode gradient × confidence (`mode_confidence`).
- N×M: 16×16,64×64,128×128,256×256,1024×1024,2048×2048,64×4096,2048×4096.
- Seeds 0–3; 20,000 optimizer updates from the same seed-specific random actor initialization: 96 runs.
- Q(a)=0.25 log[(N(a;−0.6,0.1²)+N(a;0,0.1²)+N(a;0.6,0.1²))/3], bounded a∈[−1,1]. Temperature 0.25.
- Original squashed conditional Gaussian, 1D Gaussian latent, 256×2 GELU, initial sigma 0.5, log sigma [−5,1], Adam 3e−4, proposal sigma floor 0.05.
- No EMA, clipping, learning-rate scaling, gradient norm matching or teacher change.
- Known toy basin boundaries −0.3,0.3. Teacher/assignment/confidence are detached, exactly as batch 1.

## What batch 32 means

There is still one fixed state and one frozen Q. At each optimizer update, draw 32 independent latent/candidate groups from the SAME current actor. Each group has N student samples and M candidates. Independently normalize its M teacher weights and responsibilities. Do not concatenate groups into one mixture or compute assignments across groups.

For group b, J_bij=w_bj gamma_bij, H_bmi=sum_{j in basin m}J_bij / sum_j J_bij. Select m_bi=argmax_m H_bmi and c_bi=max_m H_bmi. The routing decision remains local to the sampled component in that group.

$$g_B(\theta)=\frac1{32}\sum_{b=1}^{32}g_b(\theta),\qquad
(\theta_{t+1},\mathrm{Adam}_{t+1})=\mathrm{AdamStep}(\theta_t,\mathrm{Adam}_t,g_B).$$

This is ONE Adam update after averaging, not 32 successive updates. It has 32× candidate groups per optimizer update compared with batch 1. Thus plot optimizer steps AND wall time/sample budget; equal updates is not equal compute. H-based component routing still occurs before the shared-network VJP. Shared-parameter cross-latent interference is not eliminated.

RNG: split each update key into next_key and sample_key; derive exactly 32 group keys. All methods use the same key construction. The grouping into microbatches does not alter sampled keys. This changes the random stream relative to the batch-1 campaign, while preserving seed-specific initialization.

## Memory and numerics

Use JAX vmap within a microbatch and scan across microbatches; average gradients with frozen parameters, then update Adam once. Choose the largest divisor of 32 in {1,2,4,8,16,32} with microbatch×N×M ≤ 8,388,608. For the largest 2048×4096 matrix this is microbatch 1, still effective batch 32. Do not change N,M because of microbatching. Float32 and original matmul precision are preserved.

Validation compares microbatch 1/4/32 against an explicit 32-group gradient mean, checks exactly one optimizer step, unchanged single-group baseline gradients, checkpoint/Adam/RNG resume, H normalization, and sum of mode gradients. Each GPU worker checks all three methods on its actual GPU before training. No data collection is done on the login node.

## Measurements and figures

- 32,768 actual action samples/checkpoint; 256-bin histogram; no KDE smoothing. Analytic reference bin mass from CDF. Histogram TV = half the sum of absolute bin-mass differences.
- Same 20K update budget, evaluation milestones and checkpoints as batch 1. Store training seconds and diagnostic seconds separately.
- Each training log records average common pre-tanh marginal NLL, norm of the AVERAGED parameter gradient, mean confidence, mean retained mass, mean per-group ESS, mean per-group wmax and assigned mode fractions. In particular ESS is not computed after pooling candidates across groups.
- Diagnostics: representative first-group H in raw/z-sorted order plus all 32 H matrices; group-specific component identities are not pooled. The fixed evaluation latent bank is unchanged.
- `single_group_*` diagnostics are explicitly one-group counterfactuals. Main `gradient_gram` uses the 32-group averaged mode gradients; main `routed_*` branches perform a real 32-group Adam update from the same actor/optimizer/RNG and then discard the result. Branch histograms and fixed-latent shifts therefore match the training batch.
- Use same-seed, same-step density panels against batch 1. Do not pool incomplete/final checkpoints together as a final comparison.

## Launch and storage

Commit and push exact source/config/protocol to heejoon before launch. Archive a full hash manifest including unmodified imported batch-1 source and immutable v5 dependency snapshot. Record full commit with job IDs.

- login4 compute root: `/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/mode_gradient_batch32_20260921`
- dildata study: `/data1/heejoonorm/OptiQ/studies/20260921_gmm_mode_gradient_batch32`
- Same approved read-only collector connection, only this new child directory. No credential or permission changes.
- Up to 8 GPUs, 1 worker/GPU and 2 CPU cores/GPU; independent tasks claimed atomically. Validation dependency only. Existing MuJoCo and batch-1 work remain untouched.
