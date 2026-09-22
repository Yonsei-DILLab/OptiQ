# GMM40 oracle-energy policy amortization — 2026-09-22

## Question and objective

Which implemented actor extracts the GMM40 Boltzmann distribution accurately, with low training and single-action sampling cost?

One state, terminal reward, no learned critic: $r(a)=Q^\star(a)=\log p^\star(a)$ and $\alpha=1$. The MaxEnt objective is $E_\pi[Q^\star]+H(\pi)$, with optimum $p^\star$ on full support. This is an oracle-policy-extraction experiment, not evidence about critic learning or long-horizon RL.

Original target: DiKL GMM40, seed 0, 2 dimensions, 40 equal Gaussian components, centers uniform in [-40,40]², coordinate std softplus(1). Every learner receives an energy-only interface. Ground-truth samples and responsibilities appear only in evaluation. No supervised warm start, target initialization, or true-mode assignment is allowed.

Bounded actors use physical actions $a=50u$, $u\in[-1,1]^2$. Their mathematical optimum is the target conditioned on that box. We report its exact omitted mass and evaluate all methods against the full original target.  This small support difference is explicit; the box must not be silently shrunk to [-40,40]².

## Implemented comparisons

| Condition | Adapter | N×M | Batch | Key settings |
|---|---|---:|---:|---|
| trg64 | Current TRG Direct GMM | 64×64 | 32 | box-truncated Gaussian, log sigma [-5,-1], initial exp(-1), 256×2 GELU |
| trg256 | Current TRG Direct GMM | 256×1024 | 32 | same as trg64 |
| v5_gmm | Direct marginal GMM NLL | 256×1024 | 32 | v5 squashed Gaussian, log sigma [-5,1], initial 0.5 |
| sac | Squashed Gaussian SAC actor | — | 256 | reverse KL; alpha 1; 256×2 ReLU |
| sql | Amortized SVGD | 16 kernel particles | 256 | pinned JAX port, 8 fixed/8 updated particles; original bounded score |
| meow | MEow flow Q/V regression | — | 256 | upstream flow architecture; not SAC-NF |
| dipo | DIPO diffusion | 100 denoising steps | 256 | 20 action-improvement gradient steps; Q-driven comparator, no exact-MaxEnt claim |
| mfpo | MFPO MeanFlow | 2 sampling steps | 256 | upstream velocity/divergence learner; 16/32 teacher particles |

32 runs: 8 conditions × seeds 0–3. 100K updates each. Existing recipes retained except explicit oracle adaptation, physical scale, and batch32 for the three NLL conditions to keep the largest matrices practical. NLL batch means independent candidate clouds, not distinct states. LR 3e-4 except MEow 1e-3.

The architecture and training-compute budgets differ. These are method/recipe comparisons; the OT loss ablation was removed at user request. Plot quality against updates, measured training wall clock, and number of Q evaluations. DIPO uses Q gradients; count each action-energy point per gradient call. Initial buffer actions and uniform proposals are not ground-truth samples.

Source adapters are imported from direct-gmm-trg@36aab085bcfe8219997e1bc21e3ad8fe55f266b0. Legacy recipes are from heejoon. Upstream submodules are pinned in Git. Missing IDAC/PMOE/CPQL/SAC-NF fixed-Q adapters are not fabricated for this run. SQL is a port and DIPO/MEow/MFPO are fixed-Q adaptations, not full reproductions of original RL results.

## Metrics and figures

Evaluate at 0,1K,5K,10K,20K,50K,100K using **32,768 actual stochastic action samples**. No mu-only results, KDE smoothing, or oracle-reweighted action selection. Evaluation uses independent RNG and must not change subsequent training.

For the original target component responsibilities $r_k^\star(a)$:

$$\hat w_k=\frac1L\sum_{l=1}^L r_k^\star(a_l),\qquad C=\frac1{40}\sum_{k=1}^{40}1[\hat w_k\ge0.25/40].$$

Also report thresholds 0.1 and 0.5, posterior-mass TV against 1/40 and the fraction within 3 target standard deviations of any center. Coverage alone rewards some excessively broad distributions; do not interpret it without SWD. Forty components need not be forty isolated mathematical local maxima.

$$\mathrm{SW}_2=\left[\frac1{128L}\sum_{v,l}(\mathrm{sort}(a\cdot v)_l-\mathrm{sort}(a^\star\cdot v)_l)^2\right]^{1/2}.$$

Use 128 fixed random unit directions, physical action units, equal sample counts. Report reference-vs-reference SWD as the Monte Carlo floor. Figures A1/A2 share log single-action latency x axis and show each seed plus mean quality and seed SD. A1: coverage↑. A2: SWD↓. A Pareto-looking scatter does not establish statistically significant dominance. Supplement with native 2D action histograms, component-mass bars and training curves.

Latency is a warmed, batch1, complete action-generation call including fresh randomness and synchronized host transfer, excluding Q, OT, teacher, plotting, compilation and startup. 50 warmups; 5 blocks × 200 calls. One worker per GPU, no concurrent training on that GPU during its latency measurement. JAX JIT and PyTorch eager are disclosed. N used in training does not increase a one-action Direct GMM forward pass. Diffusion keeps all 100 sampling steps; MFPO keeps both steps.

## Validation, launch and durability

Before training: commit source/config/launcher/protocol to heejoon and record the SHA. Numerical unit checks cover posterior mass, SWD, energy-only interface, teacher stop-gradient, loss gradients, checkpoint restoration and evaluation RNG isolation. Validate original actor definitions against pinned source.

Each condition runs a 20-update GPU preflight at its actual shape. Failed methods are blocked and reported; no silent substitution. Four GPU workers then claim independent jobs. SIGTERM saves state at the next block boundary. Atomic checkpoint each 1K, full optimizer and RNG retained; single latest checkpoint per run bounds disk. 2GiB free-space guard. Store immutable code separately from outputs; back up to dildata. No credentials in archives, Git or report.

Run prepare.py once, then launch.sh in tmux. New endpoint: 38.49.42.46:60616, user heejoonorm, instance52027696, RTX5090×4. No prior experiment on a different Vast host is resumed.

## Scope correction, 2026-09-22

User requested no OT comparisons. v5 OT, legacy rowwise argmax and Monge were removed from this campaign. Their campaign-specific results, checkpoints and figures must be deleted, including backup copies. Earlier studies are outside this deletion. Original numerical source remains immutable for continuing non-OT runs; the scope amendment commit is recorded separately in deployment metadata.
