# Two-target N/M sweep with reverse score bank L=2^17

Authorized 2026-11-08 (Asia/Seoul). The user requested stopping the current
L=2^20 sweep and running L=2^17 first for the reviewer response. Preserve
old results and costs. No reviewer messages are sent by the experiment code.

## Scope and counts

- Targets: t00_reference (three equal Gaussians) and n00_spike_ramp only.
- Full Cartesian N,M in {32,128,512}, methods forward/reverse, seeds 0,1,2.
- 100,000 updates for every full run; no early performance-based pruning.
- 108 logical results = 54 FKL + 54 RKL.
- Reuse 10 completed FKL runs: six N=M128 historical baselines plus four
  N=M32 results (both targets, seeds 0 and 1) completed under ac6ed93.
- Train 98 runs: 54 fresh RKL and 44 remaining FKL. RKL N=M128 is included
  in fresh training; no L=2^20 RKL result counts as an L=2^17 result.
- All RKL starts from the original paired random initialization, including
  configurations that had partial L=2^20 training. No mixed-L schedules.
- No additional five-seed stage and no automatic L=2^20 confirmation.
- A future {32,64,128,256,512} full 5x5 expansion is reserved as a possible
  follow-up, not automatically submitted now. It adds 192 logical results
  with the same two targets, two methods, and three seeds. Decide based on
  available time/budget and report all settings, not favorable outcomes only.

## Fixed numerical settings

Actor, targets, losses, optimizer, RNG conventions, and evaluation are those
of the archived 41a020577dbf8a87d8f96d714ab3888a4bb82260 kernels. The only
changes within the archived numerical files are adding 131072 to the two
existing L allowlists; CPU validation removes that token and verifies the
original file hashes. Actual equations and accumulation code are unchanged.

- Oracle Q(a)=0.25 log target density; no learned critic; action [-10,10].
- Semi-implicit normalized truncated Gaussian actor; scalar normal latent;
  two 256-unit GELU layers; mean 10*tanh; mean initializer scale 3.
- Log sigma clipped [-5,-1], initialized -1; Adam learning rate 3e-4.
- B=32 independent groups per update. N and M are per-group counts.
- FKL uses original stopped-action, stopped-weight SNIS mixture NLL.
- RKL uses original reparameterized marginal-score gradient, independent
  L=131072 latent score bank, fresh for each group/update. The bank is shared
  across that group's M actions. Score chunk remains 4096 (32 chunks).
- Training blocks: 100 FKL updates / 5 RKL updates. Checkpoint intervals:
  1000 FKL updates / 100 RKL updates, plus completion and interruption.
- JAX 0.5.3, Flax 0.10.4, Optax 0.2.4, NumPy 2.2.6, SciPy 1.15.3.
- JAX_THREEFRY_PARTITIONABLE=false; library GEMM and
  WHILE_LOOP_UNROLLING_NO_UNROLL, matching previous runtime.
- Initial parameter hashes are checked against historical paired seeds.

## Reuse, validation, and provenance

prepare_inputs.py verifies FKL method/L, all relevant numerical and target
settings, 100K completion, initial hashes, original source commits, and
checkpoint hashes. REUSE_INPUTS.json identifies each of the ten inputs.
REUSE_AUDIT.json supplies historical initialization hashes for both methods;
its RKL records are initialization references, not reusable training results.
The ten FKL checkpoints are restored on the execution runtime and reevaluated
with 1M actual actions. Original training commits remain distinct from the
new evaluation commit.

Before production, run six true-L17 RKL pilots: both targets at diagonal
N=M32,128,512, seed 0. Each advances 20 five-update blocks (100 updates),
checks finite metrics and paired initialization, exact checkpoint contents
and RNG/counters, and one-block continuation at rtol=1e-5, atol=1e-6.
Independent evaluation must not consume training RNG. Record measured
per-update timing after compilation. FKL arithmetic is unchanged; the prior
FKL pilot evidence is retained and the ten runtime restores add verification.

The continuation tolerance reflects observed library-GEMM float32 variation
on the same GPU even without restoration; it is not a change to the optimizer.
Pilot restore validation is timed separately from pure optimizer training.
CPU tests cover the L-only guard changes, complete 108-entry coverage,
54 fresh RKL / 44 fresh FKL / 10 FKL reuse, and failure/resume cost accounting.
Production requires all six pilots and ten reuse evaluations to pass.

Commit exact sources/configuration/launch scripts/protocol to GitHub heejoon
before any new GPU trial. Deploy from that commit and verify the full source
manifest. Store job/run IDs and commit SHA with every attempt. For an unchanged
resume, keep the same commit, optimizer/RNG state, and append resume metadata.
Do not modify older frozen snapshots or automatically resume deferred L20 runs.

## Evaluation and reviewer interpretation

Use independent evaluation RNG at 0,100,1K,10K,25K,50K,75K,100K updates.
Final metrics use 2^20 actual policy samples, 512 unsmoothed bins, per-seed
histogram TV, 1D W1, and archived mode-coverage metrics. The final evaluation
sample count is unchanged; only RKL's training score bank L is reduced.

Report all grid outcomes and individual seeds / mean +/- sample SD. Select
best-in-grid by mean final per-seed TV, then W1, smaller N*M and lexicographic
order for ties. Selection and reporting use the same three seeds; this is
not held-out validation or a claim of globally optimized hyperparameters.

Clearly label the new sweep L=2^17; the main paper's original L=2^20 results
remain separately identified. Existing score-convergence evidence is from
frozen final actors and does not prove equivalent optimization trajectories
at all N/M. Compare the new N=M128 RKL runs with the existing same-seed L20
baselines as an L-sensitivity diagnostic. Report any change in TV/W1/coverage
or method ranking. Additional high-L tests are not automatically scheduled.

## Execution and storage

- Host: heejoonorm@144.34.174.130:40326; four RTX 4090 24GB GPUs.
- Base archive root: /home/heejoonorm/OptiQ/kl_nm_20261107.
- New runtime: base/revisions/l17_coarse, separate from all L20 snapshots.
- Canonical storage: dildata:/data1/heejoonorm/OptiQ/studies/20261107_kl_nm_sweep/campaign/revisions/l17_coarse.
- Four exclusive-GPU supervisor workers with a dynamic independent queue.
  No cross-target completion dependencies. Failed attempts are preserved;
  explicitly interrupted L20 tasks remain stopped.
- The preexisting restricted read-only dildata backup covers the base root
  and all revisions, every 180 seconds. Do not transfer private keys.
- Keep checkpoints, samples, logs, dependency lock, and telemetry outside Git.

## GPU accounting

Every attempt records source/config hashes, N/M/L/B, seed/phase, worker/job ID,
GPU model/UUID/count, start/end time, exit status, and resume lineage.
Monotonic worker reservations include between-task idle gaps; per-attempt
allocations include compilation, evaluation, I/O and failures. These overlap
and must not be added to each other. Queue time is not GPU execution time.
GPU utilization, device memory, and power are sampled about every five seconds.
Report sampled energy as an estimate, with coverage and missing telemetry.

Pure production optimizer training is synchronized and timed separately from
compilation, evaluation, checkpoint I/O, and pilot restore validation.
Historical pilot training timers may include restore specialization overhead;
identify them as validation costs rather than pure production throughput.
HISTORY_ROOTS.json includes the original base, revisions/v2 and revisions/v3
exactly once. All stopped L20, failed pilots, and diagnostics remain in total
campaign expenditure. Also report current L17-revision cost separately from
historical cost. Instance observation time including setup/idle is distinct
from worker allocation and is not a cloud-dollar invoice.

The monitor writes gpu_budget.csv, per-run compute_summary.json and campaign
status every minute. Forecasts use measured pilots and are explicitly estimates,
not a user-approved spending cap or completion guarantee.
