# Finite GMM: six selected Forward-KL environments

## Question and scope

Can direct gradient optimization of a finite GMM recover the six targets selected in the existing Forward/Reverse KL study? How do component count, learned variance, and equal versus learned mixture weights affect optimization?

This is a **population-objective optimization diagnostic**. It deliberately removes SNIS/candidate noise and neural parameter sharing. A finite GMM has an exact evaluable density, so there is no density-bank L. It is not a matched-compute N=M=128 RL reproduction. We do not attribute differences from the existing semi-implicit/SNIS experiments solely to infinite versus finite mixtures.

## What the cited paper actually studies

Chen, Song, Xi, Zhang, *Local Minima Structures in Gaussian Mixture Models*, §2.2–2.3, §4.3: <https://arxiv.org/pdf/2009.13040>.

The fitted density is

$$q_\mu(x)=\frac1K\sum_{i=1}^K\mathcal N(x;\mu_i,\sigma^2 I),\qquad
\mathcal L(\mu)=-\mathbb E_{p^\star}\log q_\mu(x).$$

The true mixture is equally weighted, the common variance is known and fixed, and **means are the optimized variables**. Weights and variances are not the optimization variables of that theorem. Minimizing this population NLL is equivalent to Forward KL up to the fixed target entropy.

The paper characterizes possible local-minimum structures; do not interpret the characterization as a guarantee that every such configuration is a local minimum for our numerical separation. Its 1D theorem has a very conservative explicit SNR condition that our practical target does not satisfy. Our finite numerical study is a structural diagnostic, not a verification of that theorem's sufficient SNR regime.

## Targets and run counts

The six targets are copied unchanged from `kl_six_highL_1d/config.json`:

| ID | Target |
|---|---|
| t00_reference | equal 3-GMM, centers (-4.25,0,4.25), sigma 0.5 |
| n00_spike_ramp | logistic spike plus tilted soft-edged shelf; not a GMM |
| n07_spike_flat_ramp | logistic spike, flat shelf, tilted shelf; not a GMM |
| t01_two_offset | equal 2-GMM, centers (0,4.25), sigma 0.5 |
| t05_unequal_mass | 3-GMM, masses (0.2,0.5,0.3), sigma 0.5 |
| t06_minor_mode | 3-GMM, masses (0.1,0.65,0.25), sigma 0.5 |

Main: 6 targets × K in {3,5,6,9,15} × {mean-only, mean+sigma} × seeds 0–3 = **240 runs**. All main mixtures have equal weights 1/K.

Weight control: 6 targets × K in {3,15} × learned means, sigma and softmax weights × 4 seeds = **48 runs**.

Paper control: unbounded ordinary equal-weight 3-GMM target (-4.25,0,4.25), common sigma 0.5, K=3, mean-only **plain gradient descent**, 3 initialization types × 4 seeds = **12 runs**. This isolates the paper's parameterization from box truncation and learned variance.

Total: **300 trajectories**, implemented as 75 four-seed vectorized jobs. Vectorization does not average losses or gradients across seeds.

## Main objective and parameters

On [-10,10], each fitted component is a normalized truncated Gaussian:

$$k_i(a)=\frac{\varphi((a-\mu_i)/\sigma_i)}{\sigma_i Z_i},\quad
Z_i=\Phi((10-\mu_i)/\sigma_i)-\Phi((-10-\mu_i)/\sigma_i).$$

$$q(a)=\sum_i\omega_i k_i(a),\quad
\mathcal L=-\int_{-10}^{10}p^\star(a)\log q(a)\,da.$$

| Setting | Value |
|---|---|
| Updates | 100,000, no early stopping |
| Seeds | 0,1,2,3 |
| Optimizer | Adam 3e-4; paper control GD 0.01 |
| Arithmetic | float64 throughout |
| Training integration | 2,049-point composite Simpson; normalized target masses |
| Refined validation | 8,193 points; compare final loss and gradient |
| Mean initialization | first K of 15 Uniform[-8,8] values from NumPy seed; same means for paired variance/weight conditions |
| Mean constraint | projection to [-10,10]; paper control unbounded |
| Initial sigma | 0.5 in all conditions |
| Fixed sigma | 0.5 |
| Learned sigma | optimize log sigma, project sigma to [0.05,10] |
| Equal weights | exactly 1/K, never updated |
| Learned weights | softmax of trainable logits initialized to zero |
| Checkpoint | parameters, Adam moments, step every 1K; deterministic quadrature has no training RNG |
| Intermediate evaluation | 32,768 sampled actions |
| Final evaluation | 2^20 sampled actions, 512 bins |

Means are direct free parameters; there is no mean head, tanh mean map, resampled latent, EM update, or target-component assignment in training. The finite variance cap is deliberately broader than the current semi-implicit actor's exp(-1) cap: a K=3 Gaussian must be able to represent the true width 0.5. This improves finite-GMM capacity but means the variance ranges are not matched.

The same initialization means are shared across finite variants and across targets. They are not distribution-matched to the neural actor's initial output. Any future attribution to resampling/overparameterization requires additional matched-initialization/objective ablations.

## Paper-control initializations

1. Uniform[-8,8], as in the main study.
2. Near (-2.125,4.25,4.25), with independent N(0,0.025^2) jitter: one fitted mean between two true means, two near the remaining mean.
3. Near (-4.25,0,4.25), same jitter: reference initialization near the global solution.

These deliberately structured initializations are labelled and are not pooled with random starts to report a random-start failure frequency. Integrate the ordinary Gaussian population on [-12,12]; excluded target probability is negligible and checked by quadrature normalization. Report final gradient, Hessian eigenvalues, and loss changes for 64 random directions at radii 0.001,0.01,0.1. These are finite numerical diagnostics, not proofs of local optimality.

## Measurements and interpretation

- Actual sample histograms without KDE, true target density overlaid; per-seed and seed mean.
- Histogram TV at 512 bins, integrated density TV, analytic binned TV (separates sampling error), W1 via the two exact/reference CDFs.
- Basin/core mass ratios and existing missing-mode criterion: both ratios <0.25. Peak-pass and TV<=0.15 use the existing screening definitions.
- Every component's mean, sigma, weight, and evaluation-only responsibility mass per target basin.
- Population NLL / Forward KL, gradient norm, final quadrature refinement error, active sigma bounds.

**Capacity caveats:** equal-weight K=3 or 5 cannot exactly reproduce arbitrary mode masses; even equal 2-mode targets cannot generally be represented exactly with odd K and equal narrow components. K=3,6,9,15 can represent the equal 3-GMM by duplicating means equally, but K=5 cannot reproduce it exactly. Non-GMM shelves are not exactly representable by these small finite Gaussian mixtures. Poor fit in those cases is not sufficient evidence of a bad local minimum. Report weight controls, parameter locations, and stationarity diagnostics before making optimization claims.

The selected six environments were chosen after an earlier Forward/Reverse screening. This finite comparison is conditional on that selection, not an unbiased benchmark of either KL direction or parameterization. Preserve successes and failures alike.

## Execution and provenance

Commit source, config, scripts, and this protocol on `heejoon` before GPU validation/production. Save the full SHA and SHA256 file manifest beside each immutable campaign. Use login4, one GPU and two CPUs per four-seed job, at most 12 concurrent jobs. Keep the currently running high-L experiments unchanged. Back up all outputs to dildata under `studies/20260925_finite_gmm_six/` using the existing read-only route.


### CPU execution option (2026-09-25)

The initial GPU validation waited for priority while GPU nodes were fully allocated. The CPU partition had free cores. The same float64 numerical code can therefore execute on `dell_cpu/cpu_qos`, two CPU cores per four-seed job, at most 12 jobs concurrently. Validate on the cluster CPU before launching production. This changes hardware only, not any numerical objective, parameter, seed, or update count. Store the CPU attempt under `attempt_cpu/`, preserving the unused initial GPU source and submission record.
