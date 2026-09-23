# Reverse KL, width-1 three-mode target, L=2^20

User request: compare to the completed forward KL experiment at centers (-5,0,5), target standard deviation1, N=M128,100K updates; run reverse with L=1,048,576. Reuse forward results; do not rerun or overwrite them. Seeds0–3,batch32,action[-10,10],mean initializer variance scale1,tau0.25,Adam3e-4,hidden256x256 GELU,1D Normal latent,zero state,and conditional log sigma[-5,-1] initialized-1 remain identical to forward source5b23b3a30361082936073f50804488401deaf1a4.

## Target and objective

Let f(a)=(1/3) sum_c Normal(a;c,1),c=(-5,0,5),restricted to[-10,10]. Q(a)=0.25 log f(a),p*(a)=f(a)/integral[-10,10]f. Actor is the same infinite mixture of box-truncated Gaussians, pi_theta=E_z k_theta(a|z),with physical mean10*tanh(head). There is no critic,OT,mode mask,proposal exploration injection,oracle target-sample supervision,or actor warm start. Width1 is target sigma; it does not change conditional sigma bounds.

Reverse objective:

$$J_R=\mathbb E_{a\sim\pi_\theta}[\log\pi_\theta(a)-Q(a)/\tau]+\text{constant}.$$

For each of32 independent groups of the same state,draw N128 source latents and M128 IID actions by uniformly selecting source components and drawing differentiable truncated-Gaussian noise. Independently draw L=2^20 density latents from the SAME pre-update actor. Each group has its OWN fresh density bank at EACH update; no sharing across groups,no cache,no quadrature,no subsampling of requested L. N,M control source/action sampling; L controls density-score approximation only.

$$\hat s_L(a)=\sum_{l=1}^L\frac{k_\theta(a|\tilde z_l)}{\sum_r k_\theta(a|\tilde z_r)}\frac{\mu_l-a}{\sigma_l^2},\qquad d_j=\operatorname{sg}[\hat s_L(a_j)-Q'(a_j)/\tau].$$

Backpropagate mean_j a_j*d_j only through source actions. The auxiliary density bank and direction are stopped. This is the previously validated pathwise gradient construction in kl_direction_1d; the expected explicit parameter-score term of the exact normalized infinite mixture is zero. At finite L the estimated score is a biased ratio estimator; L=2^20 does not constitute a universal accuracy guarantee. The scalar surrogate is NOT a KL. objective_estimate logs mean(log estimated_density_L-Q/tau),separately,and lacks the target normalizer constant.

## Computational implementation

Inherit the existing numerical code from kl_forward_wide_1d and enable its reverse branch. Actor,box sampler/density,optimizer initialization,source-action draws,target,and evaluation are not rewritten. Density bank is streamed in chunks4096 (256 chunks); combine sums by log-sum-exp and scores by density-weighted aggregation,never average log densities or per-chunk scores uniformly. All chunks use unchanged pre-update parameters. Chunk size differs from old128 to reduce dispatch overhead at L=2^20; IID estimator semantics are unchanged,but auxiliary random draws differ with partitioning. Same-seed initial actor/Adam/RNG and1024 evaluation samples must equal the completed forward comparator exactly.

The first GPU numerical gate at commit7ab2c09 failed the streamed-vs-dense score tolerance on very-low-density actions before the full-shape benchmark or production. The inherited recurrence reconstructed normalized chunk weights by subtracting large negative log sums,allowing their sum to deviate from one in float32. Replace ONLY this aggregation with running-max-scaled density,score-numerator,and squared-density sums. This computes the same finite-L ratio on identical chunk draws without reconstructing convex weights by log subtraction. Compare directly to dense and the prior recurrence on the same small bank. Existing forward/old reverse source snapshots are untouched.

Validation refinement: on GPU the updated accumulation still differed by0.00779 in the far-tail full-support test (reference score magnitude approximately71). Dense-vs-scan actor/pdf evaluations have different shapes/fusions. Record every action,log density,and absolute error. Retain the3e-5 scaled tolerance where the reference log density>-15; use2e-4 scaled tolerance over the full support including far tails. The direct pathwise parameter-gradient check remains unchanged at1e-4. This is a disclosed validation tolerance change,not evidence of exact floating-point equality or a guarantee about MC score convergence. If typical-region or gradient checks fail,do not launch production. GPU preflight on node33 also suffered prolonged shared-filesystem I/O and was cancelled; exclude that node for this campaign.

This evaluates32*1,048,576 additional actor latents and32*128*1,048,576 component-action pairs per update. Full shape GPU benchmarking precedes production. Do not silently lower L,batch,N,M,or total updates for speed. Keep32-bit numerical precision settings consistent with forward. Peak workspace and per-update seconds are recorded.

## Evaluation and storage

Use the identical width1 analytic reference and512 bins over[-10,10]. Save32,768 actual policy samples,unsmoothed histograms,mu/sigma,TV,W1,basin/core masses,peak criterion,backup expectation and MC standard error at the same forward evaluation steps0,100,500,1K,2K,5K,7.5K,10K,and every5K to100K. An inherited independent forward-style teacher probe is counterfactual only: prefix metrics/files counterfactual_forward to avoid implying this teacher is used for reverse training. Evaluation never advances training RNG.

Training log/stop boundary every5 updates (changed from50 for responsiveness); full actor/Adam/RNG checkpoint every100 updates and graceful signal. Per-update math is unchanged by outer logging blocks. Log wall time and compilation separately. Runs start fresh,not from benchmark updates or forward checkpoints. Preserve resume provenance and do not mix a changed numerical commit into a running source snapshot.

Validate paired initialization/config scope,target derivative,streamed-vs-dense score/density,score-vs-action-autograd,surrogate-vs-direct stopped-bank path gradient,zero auxiliary parameter gradient,and full checkpoint/RNG restoration. CPU correctness tests before deployment; committed full-shape GPU benchmark before four independently eligible production seeds. OneGPU,2CPU,32GB host RAM per seed,three-day scheduler limit,checkpoint signal10minutes before timeout. Existing/deferred jobs are untouched.

Remote:login4:/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/reverse_wide_20260923/attempt3. The parent and attempt2 retain all prior preflights and immutable sources.
Central:dildata:/data1/heejoonorm/OptiQ/studies/20260923_reverse_wide/campaign/attempt3. Backup covers the parent campaign including the preflights for provenance.
Local study:studies/20260923_reverse_wide;report:reports/20260923_reverse_wide.
Commit exact implementation/config/launch/protocol to heejoon before GPU smoke or training; record full SHA,file manifest,validation,and SlurmIDs. Reuse the existing restricted read-only backup link; no credentials in source or artifacts.
