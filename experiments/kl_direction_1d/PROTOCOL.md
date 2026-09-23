# TRG forward KL vs reverse KL: stationary symmetric 1D three modes

## Question and registered scope

With fixed per-group N=M=128, does KL direction change recovery of the same three-mode Boltzmann target? Does increasing an independent density bank L improve the reverse-KL approximation? This is actor extraction with oracle fixed Q: there is no learned critic, replay buffer, entropy backup, OT, mode-selection mask, or target sample supervision.

20 runs: forward baseline once per seed; reverse L=128,256,1024,4096; seeds 0–3. Forward is not repeated four times since it does not use L. Batch=32 independent latent/action groups of the SAME constant state, one averaged optimizer update. N and M are per group, not the total across batch. Batch32 follows the prior symmetric-three-mode study, not the later multi-landscape batch128 experiment. All 20 runs use 20,000 updates. This specifies a new experiment, with no assumed outcome.

## Exact task and reference

Actions lie in [-1,1]. Let c=(-0.6,0,0.6), h=0.1, f(a)=(1/3) sum_c Normal(a;c,h^2). Define Q(a)=0.25 log f(a) and temperature tau=0.25. The target is p*(a)=f(a)/Z on this interval, with Z=integral[-1,1] f. Use analytic normal CDFs for target histogram/basin mass and double-precision quadrature for E_p*[Q]. Q and its derivative are the only target information used during optimization. Target components/CDFs/quantiles are confined to evaluation; the analytic target score in training is exactly (dQ/da)/tau, checked against autodiff.

## TRG actor and controlled hyperparameters

The actor and normalized truncated Gaussian functions are from direct-gmm-trg commit 36aab085bcfe8219997e1bc21e3ad8fe55f266b0. See UPSTREAM.json for exact original files and hashes. The actor class and initializer were extracted unchanged; box_gaussian.py and distillation.py are byte-identical. The original distillation file includes unused OT helpers: none are invoked.

| Item | Setting |
|---|---|
| State / latent / action | 1D zero state / z~N(0,1) / action in [-1,1] |
| Actor | 256,256 hidden units; GELU; shared mean/log-std trunk |
| Conditional | Gaussian conditioned on [-1,1]; inverse-CDF sampling |
| Mean | tanh(network output); default head variance scaling 1e-4 |
| log sigma | clipped [-5,-1]; initialized -1 (sigma about0.367879) |
| Proposal std floor | exp(-5), inactive within actor range |
| N, M | 128 source/student latents, 128 actual candidate/action samples per group |
| Batch | 32 independent groups, one averaged gradient |
| Optimizer | Adam3e-4; no added gradient clipping or regularizer |
| Temperature | .25 |
| Budget | 20K updates, seeds0–3 |
| L reverse | 128,256,1024,4096 independent latent density samples |
| L chunk |128; exact log-sum-exp aggregation, no averaging of log densities |

This is the TRG actor, not v5's tanh-transformed action Gaussian. Conditional sigma is bounded at exp(-1), not exp(1). No GMM40-special mean initialization is applied. The actual action is sampled from the truncated conditional; the bounded mean is not used in its place. All methods within a seed share byte-identical initial parameters; run metadata stores their hashes. Training keys use the same draw layout; no claim of identical trajectories after policies diverge.

## Forward loss and gradient

For N fresh latents, q_N(a)=N^-1 sum_i k_theta(a|z_i). Draw M IID mixture candidates by uniformly sampling component indices (not stratified one-per-component), then conditional noise. Stop gradients through candidate actions, proposal log density and w=softmax(Q/tau-log q_N). Use the branch's existing loss:

$$L_F=-\sum_j \operatorname{sg}(w_j)\log\left[\frac1N\sum_i k_\theta(\operatorname{sg}(b_j)\mid z_i)\right].$$

Only the student conditional parameters, including the differentiable truncation normalizer, receive gradients. This is a finite-N density and finite-M self-normalized importance approximation to forward KL; it is not an exact unbiased KL estimator.

## Reverse path gradient and L

The infinite-mixture policy remains pi_theta(a)=E_z k_theta(a|z). Reverse KL is E_{a~pi_theta}[log pi_theta(a)-Q(a)/tau]+log Z_Q. With normalized fixed-support conditionals, the expected explicit parameter score term vanishes, leaving

$$\nabla_\theta D_{KL}(\pi_\theta\|p^*)=\mathbb E[J_\theta(a)^\top(\nabla_a\log\pi_\theta(a)-\nabla_aQ(a)/\tau)].$$

Draw the same N source latents and M uniform-component conditional samples; these source actions remain differentiable. Independently draw L auxiliary latents from the SAME pre-update actor. Stop gradients through auxiliary conditional parameters. With gamma_l(a)=k_l(a)/sum_k k_k(a),

$$\widehat s_L(a)=\sum_{l=1}^L\gamma_l(a)\frac{\mu_l-a}{\sigma_l^2}.$$

The truncated normalizer does not depend on interior action a, but is included when calculating the responsibilities. Construct the stopped direction d=sg(s_L(a)-Q'(a)/tau) and backpropagate the surrogate mean_j a_j d_j. This propagates only through source actions. The surrogate's scalar value is NOT a KL estimate. A separately logged mean(log density_L-Q/tau) is also a biased diagnostic (KL minus the fixed normalizer constant), not a ground truth KL.

The L-bank is independent of source components and noise; source components are not forced into it. Chunk contributions are pooled using log-sum-exp and density-proportional score weights, all at unchanged theta. There is no stale density cache or parameter update between chunks. L changes density estimation compute, not actor architecture, N, M, or inference procedure. The score is a finite-sample ratio estimator and remains biased. Increasing L controls a confound; it does not guarantee monotonic learning improvement. N+L actor forwards and Q gradients make reverse more expensive, so update-based and training-time-based results are both required.

## Measurements and figures

At updates 0,100,500,1000,2000,5000,10000,20000, draw 32,768 actual policy actions using an evaluation-only RNG; save raw actions, their mu/sigma, and unsmoothed 256-bin histograms. Plot exact target density as dashed black and sampled policy histogram as a step curve. No analytic policy integration or KDE smoothing is used for policy plots. Evaluate histogram TV=(1/2)sum_bins|estimated mass-target bin mass|, 1D W1 via analytic target quantiles, and backup error=mean Q(policy sample)-E_p*[Q]. Record backup Monte Carlo standard error.

Basin/mode masses use fixed intervals [-1,-.3),[-.3,.3),[.3,1]. Basin TV compares these three probabilities to exact target interval masses; it does not measure within-basin shape. A mode counts as covered at >=25% of its target basin mass, so coverage alone cannot show accurate density recovery. Report sample TV alongside mode masses. Sigma, between-mean variance and within-conditional variance give context, not causal proof.

At updates0,2K,20K, freeze actor and 128 probe actions; estimate action scores for all four L with three independent repeated banks. Compare to an independent Lref=16384 MC bank, and additionally double that bank to32768 to expose reference instability. Reference is also approximate: residual checks must be shown. The same small-bank seed gives nested prefixes across L. Record score RMSE, relative RMSE, repetition spread, log-density error and effective contributing component count. Raw scores are saved. No diagnostic gradients update the actor or advance the training RNG.

Report paired seeds, aggregate mean±SD, all individual histograms, TV vs update and training wall time. Do not compare raw forward and reverse objective values as a common metric. JIT compilation and evaluation/diagnostic time are separate from synchronized training wall time.

## Validation, provenance, storage and scheduling

Before training: verify upstream hashes, target normalization and derivative, dense-vs-chunk score, score-vs-action-autograd, reverse surrogate parameter gradients vs direct detached-bank pathwise gradients, forward gradients vs explicit marginal NLL, zero auxiliary parameter gradient, paired initialization, complete checkpoint resume and evaluation RNG isolation. Then run full-shape GPU smoke for all five conditions, checking finite/nonzero gradients and sigma bounds. Any validation dependency is solely a correctness gate. All20 production runs become independently eligible afterwards.

Commit exact files/config/protocol to heejoon before optimization or GPU smoke. Immutable git archive, SHA256 file manifest, source commit, upstream commit and job IDs accompany the deployment. Use login4 SLURM, oneGPU and2CPUs/job, up to8 concurrent runs,8h limit. Exclude previously malfunctioning CUDA nodes. SIGUSR1/TERM requests a full actor/Adam/RNG checkpoint at the next50-update boundary. Save every500 updates. Resume is explicit, with unchanged implementation SHA. No silent retries or old-run overwrites.

Study root on login4: /scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/kl_direction_1d_20260923.
Central storage: dildata:/data1/heejoonorm/OptiQ/studies/20260923_kl_direction_1d/campaign. Existing folder-restricted read-only backup key covers this child; sync every180seconds, no deletion. Credentials are never included in the source or results. Local protocol/execution notes: studies/20260923_kl_direction_1d. Reports: reports/20260923_kl_direction_1d. No MuJoCo or existing jobs are modified.
