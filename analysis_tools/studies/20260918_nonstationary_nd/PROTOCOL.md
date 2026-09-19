# Non-stationary Q: dimension, separated modes, and Sinkhorn epsilon

2026-09-18 user request. Main question: after Q changes, which algorithm adapts
fastest while actually reproducing the new distribution? Preserve prior Q tasks,
add an easier separated two-mode task and dimensions2,4,8. The only historical
MSE baseline retained is the original deterministic row-argmax with truncated
Gaussian KDE. No barycentric, categorical, or unbounded/clipped Gaussian MSE.

## Registered design

All GPU matrix multiplications use `JAX_DEFAULT_MATMUL_PRECISION=highest`.
This makes the forward/backward precision explicit across different GPU models
and independently compiled legacy-update verification paths. Parameters remain
float32; exact OT and analytic references use float64 on the CPU.

- Dimensions1,2,4,8; new N×M256×1024,512×512 (completed16×64 results retained externally); seeds0–3.
- Temperature0.25; Adam3e-4; no clipping/EMA/history/sigma annealing.
- Seven original NLL methods are retained. Learned-sigma Sinkhorn expands to
  epsilon `[0.0001,0.001,0.01,0.1,1,10]`.
  Fixed-sigma0.1 and0.5 Sinkhorn retain epsilon0.1. Including historical argmax,
  there are13 methods. The user explicitly listed six epsilon values; these take precedence over the word five.
- Exact and Sinkhorn learned sigma use v5 conditional Gaussian actor256×2 GELU;
  log sigma clip[-5,1],initial sigma0.5. Fixed sigma overrides EVERY sampling
  and fitting path. Proposals: conditional Gaussian mixture IID candidates,
  floor0.05, importance weights softmax(Q/0.25-log q), no anchors.
- NLL cost is RAW sum of squared action-coordinate distances. Epsilon is in
  these cost units; do not divide by dimension or average cost. Record cost
  scale and epsilon alongside solver marginal error to interpret dimensions.
- Sinkhorn100iterations, original log-domain implementation and subsequent
  row normalization. Very small epsilon can remain unconverged: raw P row/col
  residual, effective R/N column residual, zero rows, and timing are recorded.
  Do not call an unconverged result a converged entropic OT solution.
- True Exact OT: original sorted monotone coupling only in1D; full squared
  Euclidean network-simplex EMD/POT inD>1, CPUfloat64 solver with strict marginal
  checks and nonconvergence errors. Float32 plan enters original v5 loss.
  Never substitute coordinate-sorted1D transport for multidimensional OT.
- Legacy baseline: archived pre-v5 ImplicitActor256×3,a=clip(G(s,z),-1,1),
  historical update_actor directly called. KDEstd0.2,local clip0.5, stratified
  N×(M/N) candidates, no anchors; mean-normalized cost, epsilon0.05,30iterations,
  density correction1, rowwise argmax MSE. This is an algorithm comparison,
  not an architecture/proposal-matched ablation against NLL.

## Analytic targets

Actions a in[-1,1]^D. Only coordinate1 carries the separated modes; coordinates
2..D are independent N(0,0.35²), truncated by the action box. Thus the entire
joint distribution has exactlyK modes rather than K^D combinations. This is a
controlled easy extension; it does not represent arbitrarily correlated targets.

\[
f_t(a)=\left[\sum_k\rho_{kt}{\cal N}(a_1;c_{kt},h^2)\right]
\prod_{d=2}^D{\cal N}(a_d;0,0.35^2),\quad
Q_t(a)=0.25\log f_t(a),\quad \pi_t^*(a)=f_t(a)/\int_{[-1,1]^D}f_t.
\]

1. **double/mass:** centers(-0.65,+0.65),width0.12. Updates1..20K use weights
   (0.5,0.5);20,001..25K use(0.8,0.2);25,001..30K use(0.2,0.8);30,001..35K
   return to(0.5,0.5). The center distance1.30 is10.83 conditional standard
   deviations. Both modes remain present throughout.
2. **tri/mass:** original centers(-0.6,0,0.6),width0.1, uniform20K prefix,
   then(0.6,0.2,0.2),(0.2,0.2,0.6),uniform at identical5K boundaries.
3. **tri/split:** original20K prefix, d increases0→0.15 over20K..22K, holds
   through25K, decreases to0 at27K,then holds through35K. Centers c_k±d,
   six equal Gaussian components. Exactly3→6→3 joint peaks.

Each method/size/dimension/seed learns its OWN20K prefix from initialization.
Tri mass/split share that exact actor, Adam, RNG checkpoint. No common trained
actor is forced across algorithms; poor initial fits remain visible. Present
pre-change accuracy alongside adaptation speed and report unattained thresholds
as censored rather than as slow-but-successful convergence.

## Learned-Q tasks (tri family, repeated in every dimension)

State/action in[-1,1]^D,s'=a,s0=0,200-step timeout resets state but preserves TD
bootstrap. Reward
\[
g(a)=\sum_{c\in\{-0.6,0,0.6\}}e^{-(a_1-c)^2/(2\cdot0.1^2)}
\prod_{d=2}^D e^{-a_d^2/(2\cdot0.35^2)},\quad
r(s,a)=g(a)-0.25 D^{-1}\|a-s\|_2^2.
\]
D=1 exactly matches the prior MDP. Mean squared movement cost keeps its scale
comparable as coordinates are added. Reward/transition remain stationary; Q and
policy evolve through learning. Use v5 reward-only twin TD: target min,
gamma0.99,target coefficient0.005,replay batch256,5K uniform exploration then
35K learning steps, UTD1. Live twin mean for extraction.

- **source:**4seeds×4dims,learned-sigma Sinkhorn epsilon0.1,N16/M64,actor state
  batch256. Store the complete live critic parameters AFTER EACH critic update.
- **replay:** all methods receive the SAME source critic at each update and
  query Q_t(0,a). No action interpolation or temporal thinning. Relative to the
  old1D grid replay this uses the full network; it is explicitly documented.
- **closed:** each method collects its own data and learns its own critic,
  actor state batch1,critic256, as in the previous diagnostic experiment.
- D>1 learned-Q references use a FIXED independent scrambled Sobol uniform
  quadrature(131072 points), highest-precision Q evaluation, normalized exp(Q/tau)
  weights. Report ESS and discrepancy between the two halves; ESS<512 or maximum
  marginal half-TV>0.05 marks the reference unreliable. No claim of exact joint
  reference under those failures. D1 uses8193-point integration. Analytic targets
  have exact factorized reference integrals and direct exact target samples.

## Tracking speed and figures

-32768 actual actor samples at each diagnostic time;512fixed bins/coordinate.
 No KDE/CDF integration to make actor curves. True analytic target uses exact
 bin integrals. Target and actor always share bins.
-Evaluate every200updates;20updates within100 of change boundaries; capture
 pre/post-change snapshots. Keep full32768samples at selected steps, not at every
 evaluation. Preserve independent final samples and fixed128evaluation latents.
-Primary: first-axis histogram TV (direct continuity with1D), mean/worst marginal
 TV,2D joint64×64 histogram TV, joint sliced Wasserstein-1(32fixed projections,
 4096actual samples). Marginal TV is never labelled full joint TV in4D/8D.
-Record mode/basin mass,backup bias,proposal/weighted-teacher/actor discrepancies,
 ESS,wmax,sigma,component usage, and OT marginal errors.
-Record cumulative **training compute seconds** separately from end-to-end
 elapsed time including diagnosis/I/O. Timing compares hardware strata; do not
 call mixed-GPU wall clocks algorithm-only speedups. Update-count plots remain
 primary for causal comparison. Solver CPU time is part of Exact OT training.
-Post-shift adaptation: error curves,AUC over first100/500/1000/5000updates;
 first stable time reaching absolute TV0.1 and pre-change TV+0.05 (three
 consecutive evaluations). Report censored runs, final accuracy, and the
 plateau-adjusted excess error rather than declaring a broad fast fit successful.
-D2:3D target density surface and actual sample2D histogram surface, plus top-down
 heatmaps; vertical axis is density, not an action coordinate.
-D4/8:coordinate1 marginal and other coordinates separately, plus per-coordinate
 panels and fixed-latent mean/sigma evolution. Do not pool different marginals
 into one ambiguous curve.
-Assignment panels: original and action1-sorted matrix; GMM w_j gamma_ij clearly
 labelled effective assignment, not OT. Sorting by action1 is only a display
 order inD>1, not a solver or a claim of scalar geometry.
-Sweep plots:6epsilon small multiples, backed by GMM/Exact/legacy anchors;
 full13-method spaghetti plots are avoided.

## Queue, validation, storage

13methods×4dims×2 NEW sizes×4seeds=416cells. Five final trials percell =2080 new comparisons. Prefixes832 =>2912 newly executable tasks. Reuse16 completed common-Q sources through explicit IMPORTED_Q_SOURCES.json provenance; never relabel their commit. The retained16×64 subset contributes1040 completed comparisons, so the reduced scientific scope is3120 comparisons. All1823 completed old-scope final comparisons (including removed sizes/eps) remain in the retrospective report; they are not all part of the reduced scope.

All unfinished old-scope trials have been checkpointed/stopped. No old large-size checkpoint initializes a smaller matrix experiment. New sizes start their own actors/Adam/RNG at update0. An original run in an excluded scope is never deleted or restarted by this queue. Prioritize immediately eligible mass/split after a prefix, then common-Q replay and additional prefixes;18 fast workers plus2 multidimensional Exact workers initially,2CPUs each. No performance-based selection.

Only per-trial prefix and same-seed source dependencies exist; no global stage
or environment completion chains. Multiple workers dynamically claim ready work.

Validation before main: target normalization/peak counts, exact OT vs independent
LP, all13methods finite at small size, representative methods at all large sizes,
fixed sigma invariance, GMM forbidden OT calls, archived legacy production update
matching, full actor/critic/replay/RNG checkpoint continuation, source B256 and
full critic replay roundtrip, diagnostic file validity. Time large Exact OT
solves explicitly; expose high runtime in validation and progress estimates.

Commit exact source/config/tasks/protocol before any GPU validation or main
launch on heejoon. Frozen revision directory prevents edits to active sources.
Checkpoints every200updates, full Adam/replay/critic/target/RNG/env saved atomically.
Up to16GPUs initially,CPU2/GPU,RAM32GB,14hour slices with checkpoint/requeue on
advance time-limit signal. Failed/preempted lease can be reclaimed by remaining
workers; numerical failures are recorded and not endlessly retried. If all workers
end, submitting another unchanged worker wave retains original commit provenance.

login4 root: /lustre/hobbit9882/OptiQ-nonstationary-q-20260917/extensions/20260918_nd/REVISION.
dildata root: /data1/heejoonorm/OptiQ/studies/20260918_nonstationary_nd/REVISION.
Reuse the original study's READ-ONLY restricted backup connection; credentials
never enter source or reports. Source/manifests,logs,full checkpoints,data live on
central storage. Original experiments and MuJoCo runs remain unmodified.

## September19 legacy TD correction

The first wave's legacy closed-loop path incorrectly sent zero TD perturbation
std/clip through a truncated Gaussian CDF, producing NaNs on the first critic
update. In the corrected immutable revision, zero std OR zero clip bypasses
that sampler and returns the already-sampled legacy action unchanged. Latent
sampling, twin TD objective, RNG split, actor fitting, target EMA and all other
hyperparameters are unchanged. Conditional Gaussian TD does not enter this
branch and is unchanged.

Historically, the corrected revision was used ONLY for64 legacy closed-loop trials (16 failed
N16/M64 replacements plus48 not-yet-started larger trials). Completed analytic,
replayed-Q and NLL results and their checkpoints stay under the original
cc11f537af330e23e1cc77cb94a9426660b55ebb revision. Validate independent TD
target/loss/gradient and finite actual closed-loop checkpoint continuation in
all four dimensions before registering corrected production runs.

## Scope reduction / provenance (September19)

The new snapshot incorporates the already-validated zero-TD-noise correction. All new tasks, including legacy closed-loop, use this corrected source. Original cc11f537/91cb9c3 snapshots remain immutable. Imported source critic trajectories retain cc11f537 commit and source IDs, checked against COMPLETE and ARTIFACTS_SHA256 manifest hashes. Streams are read-only symlinks to original storage; import validation runs actual replay updates. New actor checkpoints retain only the new consumer code ID. Q_SOURCE.json records both source and consumer IDs. No source critic retraining or new16×64 comparison is scheduled.

The two new matrices both contain262144 entries (1/16 of a4194304-entry matrix). Computation is not assumed to scale linearly: solver/JIT/diagnostic timings remain measured. Temperature, architectures, schedules, updates, seeds and solver iteration limits are unchanged.
