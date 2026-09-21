# Frozen Q: Direct GMM vs mode selection, batch 128

2026-09-21. User requested the previously created frozen-Q landscapes, no confidence weighting, N×M=16×16 / 64×64 / 64×256 / 128×128 / 128×256 / 512×512 / 512×2048, login4. Four paired seeds (0–3). Three landscapes × seven sizes × two methods × four seeds = **168 runs**, 20,000 updates each.

## Scientific question

Does keeping only each sampled latent's winning-mode contribution improve marginal distribution fitting across narrow, unequal and numerous modes? This is an **oracle basin-label toy ablation**: known frozen-Q basins provide candidate labels. It does not implement automatic mode discovery for MuJoCo. Mode selection is a modified gradient; it is not the exact marginal NLL gradient and does not eliminate all interference through shared network parameters.

## Original Q functions and references

`legacy_problems.py` is byte-identical to `studies/20260915_1d_landscapes/experiment/problems.py`, SHA256 `7b47e7835cc14155caf456f31e7473511dcd5ccc238d608be57b5aaaf80a2328`. `legacy_reference_fixture.json` preserves references extracted from the original completed runs. No change to Q or mode widths is introduced.

$$f(a)=\sum_k\rho_k\mathcal N(a;c_k,h_k^2),\quad Q(a)=0.25\log f(a),\quad p^*(a)=f(a)/\int_{-1}^{1}f(x)dx.$$

| Q | Centers | Standard deviations | Mixture weights |
|---|---|---|---|
| needle3 | −.7, 0, .7 | .035, .18, .06 | .3, .4, .3 |
| comb6 | −.75, −.45, −.15, .15, .45, .75 | .035 each | 1/6 each |
| rugged7 | −.9, −.6, −.32, .02, .3, .58, .9 | .025, .075, .02, .11, .035, .06, .018 | .12, .18, .10, .20, .12, .16, .12 |

Each basin is the interval between adjacent density valleys. Basin masses are integrated from the normalized Gaussian-mixture CDF; nominal component weights need not equal basin masses. Exact target Q expectation is checked by adaptive and fine-midpoint integration.

## Actor and teacher, common to both methods

Reuse the immutable 0917 toy actor/update implementation imported by `gmm_gradient_interference`, including its own v5 modules, not the recent TRG MuJoCo actor. One constant state, one-dimensional standard Normal latent; 256×2 GELU network; learned pre-tanh mean/log standard deviation, log sigma clipped to [−5,1], sigma initialization .5; Adam learning rate 3e−4. No EMA or gradient clipping. Fresh random initialization (not a pre-fitted multimodal actor). Same seed produces identical initial parameters and paired RNG streams.

$$a=\tanh(\mu_\theta(z)+\sigma_\theta(z)\epsilon).$$

Each group has N newly sampled latent components and M proposal candidates. The existing conditional Gaussian-mixture proposal, sigma floor .05, component sampling, density correction and RNG path are unchanged; only the frozen Q is substituted. Temperature is **.25**.

$$w_j=\operatorname{softmax}_j[Q(b_j)/.25-\log q_F(b_j)],\quad \gamma_{ij}=\frac{k_i(b_j)}{\sum_\ell k_\ell(b_j)},\quad J_{ij}=w_j\gamma_{ij}.$$

Teacher candidates, weights and routing coefficients are detached. Pre-tanh NLL and action-space NLL have the same actor gradient because the target Jacobian is parameter-independent. Logged common NLL is pre-tanh.

## The only experimental change: gradient routing

$$\alpha_i=\sum_jJ_{ij},\quad H_{mi}=\frac{\sum_{j:b_j\in B_m}J_{ij}}{\alpha_i},\quad m_i=\arg\max_m H_{mi}.$$

Baseline differentiates the original marginal GMM NLL. Mode selection uses

$$L_{\rm route}=-\sum_{ij}\operatorname{stopgrad}\!\left[J_{ij}\mathbf1\{b_j\in B_{m_i}\}\right]\log k_i(b_j).$$

This gives exactly the selected mode's output gradient, propagated through the common actor network. **No confidence multiplication, no renormalization, no extra 1/N**. Ties use first argmax; zero-usage rows have zero retained coefficient. Confidence is logged only as a diagnostic. Since selection removes terms, its gradient scale may also change: this is the requested unchanged mode-only variant, not a norm-matched ablation.

## Batch 128 and compute controls

An update draws **128 independent groups at the same pre-update parameters**, averages their gradients, then performs **one Adam step**. It is neither a mixture of 128N components nor 128 sequential updates. For memory, gradients are accumulated over microbatches with ≤8,388,608 pairs; 512×2048 uses microbatch 8. All keys are drawn in advance, so the mathematical batch and RNG are independent of microbatch size. Validation checks explicit batch means and multiple microbatch sizes.

Twenty thousand updates thus use 2.56 million independent groups per run. This costs substantially more than batch-1 at equal update count. Report both optimizer steps and training wall time; report diagnostics time separately. Use 12 independent Slurm workers, one GPU and two CPUs each, on allowed partitions. Only validation gates submission; experiments have no inter-run completion dependencies. Existing experiments remain untouched.

## Measurements and figures

- Primary density: **32,768 fresh actual actions**, 512 equal bins in [−1,1], no KDE/smoothing/integration of actor density. Exact target curve/CDF can be integrated. Store full samples and all four seeds.
- Histogram TV: half the sum of absolute actor-vs-target bin-mass differences.
- Basin TV: the analogous difference across valley-defined basins; measures allocation, not within-mode fitting. Coverage: fraction of basins with ≥25% of target basin mass; threshold stated on figures.
- Backup error: sample mean Q minus numerically verified target expectation.
- Fixed evaluation latents (2048) for mean/sigma and conditional basin mass. ≥.8 conditional basin probability identifies a specialist; other components are called mixed.
- Each diagnostic stores representative group-0 proposal/candidates, Q, density, logits, weights, effective GMM assignment J and normalized H. Draw both original-order and sorted heatmaps, labelled as **GMM responsibilities, not OT**.
- Also store all128groups' H, usages and teacher basin masses. Mode gradient Gram/cosines average all128groups; paired counterfactual baseline vs selection uses exactly the same saved actor/Adam state and random keys and is discarded after measurement.
- Evaluate at 0,1,10,100,500,1000,2000 and every1000 thereafter. Detailed diagnostics at 0,1,10,100,500,1000,2000,5000,10000,15000,20000.
- Atomic checkpoints (parameters, Adam state, RNG, commit) every500updates, at evaluation milestones and graceful signal exit. Complete flags only at20K. Do not equate early interruption with final performance.

## Reproducibility and validation

Before launch, commit exact source/config/protocol/launch scripts to GitHub branch `heejoon`, push, freeze files from that commit, and store SHA beside manifest and Slurm IDs. Validate references, unchanged proposal path, exact baseline gradients, selected-output-gradient/VJP equivalence, no confidence scaling, stopped routing gradients, batch mean, one Adam step, serialization/RNG resume and generalized 3/6/7-mode diagnostics. CPU validation followed by worker GPU smoke tests. Sync immutable source, logs, checkpoints and results into `dildata:/data1/heejoonorm/OptiQ/studies/20260921_gmm_mode_selection_frozen`, outside Git.
