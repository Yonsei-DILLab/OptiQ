# GMM40: mode allocation and within-mode accuracy

## Question and provenance

Compare legacy rowwise argmax, genuine finite empirical 2D Monge, v5 conditional
OT NLL, and Direct GMM marginal NLL. Mode seeking, mode mass allocation, and
within-mode shape are separate endpoints. No expected winner is assumed.

GitHub references are pinned in `references/BRANCHES.json`:
- `heechan-no-anchor@1d9394ad7e35d18856115f47983d19f81e4f136e`:
  successful legacy HP19/replication base (50K) and common T1 polish (75K).
- `heechan@c6132fd94ec0981cb47e9b25c936ff571abf9d5b`: no v5 GMM40 code/config found.
- `v5-gmm40@a3b1ffc272e54119d6ea9752ace105be47178800`: initialized from v5,
  but no GMM40 benchmark/tuned recipe in the branch. **We do not claim to reproduce
  a successful v5 GMM40 recipe.** We use its actual actor, conditional proposal,
  mean-action OT, NLL, and default epsilon/iterations in a documented adapter.

The extracted actor/conditional-proposal/NLL definitions are unchanged copies;
source branch IDs and original configuration files are retained. Existing root
legacy actor/update files are unchanged. Numerical validation compares the new
legacy path against the original `benchmarks.gmm40.sampler.update`.

## Target and coordinates

Official existing target: 2D, 40 equally weighted diagonal Gaussians. Centers use
Torch seed 0, uniform on [-40,40]^2; each coordinate sigma is softplus(1)=1.31326.
All seeds share this target. Q is exactly log p_GMM40; there is no learned critic.
Temperature=1 and density beta=1 recover the original GMM as the ideal IS target.
Using MuJoCo T=.25 on log p would instead target p^4, so we do not do that.
No target samples or target component labels are used for training.

Legacy/Monge use native unbounded x=G(z). v5/GMM use x=50 tanh(mu+sigma*epsilon),
with normalized action a=x/50 and pre-tanh u retained. Their density includes
both the tanh Jacobian and the physical-coordinate constant -2 log(50).
Target probability outside [-50,50]^2 is negligible (<1e-12); legacy remains
unbounded. This adapter keeps the actual v5 squashing and sigma range.

## Conditions (32 runs)

| Condition | N x M | Hidden layers | Training |
|---|---|---|---|
| legacy_reference | 2048 x 2048 | 512 x 5 | original successful 75K recipe |
| legacy_matched | 256 x 1024 | 512 x 5 | same recipe with smaller N, 4 proposals/center |
| monge_matched | 256 x 1024 | 512 x 5 | identical legacy setup, assignment replaced |
| v5_ot_matched | 256 x 1024 | 256 x 2 | actual v5 actor/proposal/OT-NLL |
| gmm_matched | 256 x 1024 | 256 x 2 | same as v5; loss-only change |
| gmm_small | 16 x 64 | 256 x 2 | Direct GMM count ablation |
| gmm_large | 2048 x 2048 | 256 x 2 | Direct GMM count ablation |
| v5_ot_large | 2048 x 2048 | 256 x 2 | large-N loss-only v5/GMM pair |

All conditions: seeds 0,1,2,3, state batch=1, Adam (0.9,0.999), lr3e-4 through
50K and 1e-4 thereafter through75K. Optimizer history is retained at50K.
Exact source/config/protocol committed to `heejoon` BEFORE validation or runs.
Every segment references the immutable full commit and source manifest.

Legacy/Monge: randomized stratified Gaussian latent grid; unbounded Gaussian KDE;
no anchors; physical proposal std8->1 exponentially over15K, then1; squared cost
normalized by its mean; Sinkhorn epsilon .01->1e-4 over15K,300 iterations.
Legacy loss is action-space row-argmax MSE. No initial broadening/calibration.

v5/GMM: IID Gaussian latent; initial sigma=.5, log sigma clipped[-5,1]; initial
mean head scale1e-4, sigma head kernel0; conditional Gaussian teacher with pre-tanh
sigma floor .05, IID mixture-component candidate sampling; v5 source=tanh(mu),
squared normalized-action cost (NOT mean-normalized), Sinkhorn epsilon=.1,
100 iterations, row-normalized conditional Gaussian NLL. GMM uses the same
weighted teacher and heads, but `logsumexp_i(log k_i)-log N` directly, without OT.
Jacobian constants need not enter the student loss because teacher is stopped.

**Comparability:** v5 OT versus GMM at a common size isolates loss/assignment.
Legacy versus Monge isolates assignment/teacher quantization. Legacy versus v5
compares practical recipes with different networks, priors/sampling, proposal,
coordinates and cost scaling; do not call this a controlled loss ablation.

## Monge definition and limitation

The original weighted M-point teacher has arbitrary masses and is generally not
realizable by a deterministic map from N equal-mass particles. Randomly permute
candidates, then systematic-resample N representatives using one uniform offset.
Solve the N x N squared-Euclidean cost with SciPy linear_sum_assignment. This is
the exact optimal bijection for those empirical equal-weight measures, not exact
transport to the original M weighted atoms. Repeated representative coordinates
are allowed; the indexed assignment is one-to-one. Record quantization atom-TV,
teacher mode mass, selected-target mode mass and cost. This is NOT 1D sorting and
NOT Sinkhorn argmax relabeled Monge. Validate small 2D problems by brute force.

## Evaluation and visualization

At0 and each5K:32,768 IID *raw* actor actions; no KDE smoothing, rejection, or
importance correction of actor evaluation samples. Evaluation RNG is independent
from training; fixed probe seed, common fixed target reference. Reference-vs-
independent-reference metrics give a finite-sample floor.

1. Posterior responsibility under the exact target gives all40 mode masses.
   Mode-mass TV = .5 sum_k |mass_k-1/40|. Report nearest/core behavior separately:
   coverage requires posterior mass within3sigma >=25% of nominal mode mass.
2. Within-mode centers/covariances are responsibility-weighted, compared to target
   centers/covariances; avoid arbitrary nearest-mode truncation. Missing modes
   remain missing; average shape diagnostics only over mass>=10% nominal and
   always report that mode count. Retain full40-mode statistics.
3. Sliced W2 (32 fixed directions), mean logp and histogram TV (160x160 physical
   bins[-50,50]^2 plus outside bin). Mean logp alone is not distribution fidelity.
4. Independent teacher probe: candidates, Q, proposal log-density, weights, ESS,
   mode masses; separate teacher error from actor/teacher mode discrepancy.
5. Sinkhorn finite-iteration row/column residual and effective column error after
   row normalization. GMM effective plan=w_j*posterior responsibility. Usage ESS
   is not interpreted as permanent latent identities.
6. At0,15K,30K,50K,75K retain full assignment matrices, raw AND mode/x1-sorted
   heatmaps (display block sums if >256). Raw actor/target/teacher 2D histograms,
   mode-mass bars, fixed components0/13 close-ups. All40 shape statistics retained;
   these illustrative components are selected before results.

Both update/query counts and actual training seconds are recorded. Training time
includes compilation and, for Monge, CPU assignment/transfers. Evaluation time
is excluded from training seconds; total segment wall time is recorded separately.
75K versus successful legacy parent budgets are comparable, not short-screen
results substituted for mature runs.

## Execution and storage

Server heejoonorm@31.148.50.247:11717, GPU3 only; CPU6–9. Existing MuJoCo GPU0–2
workers untouched. One tmux worker rotates15K-update checkpoint segments over
conditions and seeds, then repeats to75K. No sample/seed filtering; failures stay
visible. Latest checkpoints contain actor, optimizer, RNG, step and source SHA.
W&B project `OptiQ/GMM40_heejoon`, one stable ID per run, resumed segments share ID.
API credential loaded from the already authorized external0600 secret file;
never copied into code, logs or backup.

Full study under `/home/heejoonorm/OptiQ/legacy_monge/gmm40/<commit>/` so the existing
dildata collector retrieves it every120s to
`/data1/heejoonorm/OptiQ/studies/20260919_legacy_monge/remote_3114850247/gmm40/<commit>/`.
No result is claimed at launch. Preflight checks legacy parity, paired v5/GMM
teacher identity, loss/gradient formulas, no OT call for GMM, assignment optimality,
finite training and per-condition runtime on GPU3.
