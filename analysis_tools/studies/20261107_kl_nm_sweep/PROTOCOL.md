# Main-paper two-target FKL/RKL N–M sensitivity study

Status: authorized for execution on 2026-11-07; commit and preflight precede production training.
Date: 2026-11-07 (Asia/Seoul).

Revision: the user narrowed the scope to the two main-paper Figure 2 targets
and requested actual GPU-budget accounting. The earlier five-target,
13-point sparse design is superseded by the regular 3x3 proposal below.
The user subsequently removed the additional five-seed confirmation stage;
only the three-seed sweep is planned.
The user authorized baseline-reuse verification and execution of remaining
runs on heejoonorm@144.34.174.130:40326, with a pre-run heejoon commit,
GPU accounting, and dildata storage.

## Question and scope

Test whether the forward/reverse KL comparison reported in Section 5.1,
Figure 2 of submission 30820 depends on the
shared choice N=M=128. Give reverse KL the same explicit N/M search space
and number of tuning trials as forward KL. Report the full three-seed grid;
there is no separate fresh-seed confirmation stage. Do not interpret this
study as optimization of every baseline
hyperparameter or evidence about targets outside the two main-paper cases.

The paper source is:
`/Users/heejoon/Documents/yonsei/DIL Lab/30820_Efficient_Implicit_Boltz.pdf`.

Only these two targets are eligible, in manuscript Figure 2 order:

| ID | Reported target |
|---|---|
| t00_reference | Three equal Gaussian modes at -4.25, 0, 4.25; SD 0.5 |
| n00_spike_ramp | Logistic spike plus rising ramp, masses 0.2 / 0.8 |

Reuse their exact archived target implementations and parameters after
matching them to Appendix C.2. Do not select new targets based on outcomes.
Appendix-only targets n07_spike_flat_ramp, t01_two_offset, and
t05_unequal_mass, the historical sixth target t06_minor_mode, GMM40, and
control tasks are outside this experiment.

## Evidence checked while planning

The archived 20260923_forward_sizes study did run N=M=256 and N=M=512,
alongside an existing N=M=128 baseline. It was forward-only, with seeds
0–3, action range [-1,1], Gaussian centers (-0.6,0,0.6), SD 0.1,
mean-head initializer scale 1e-4, and 20K updates. Those runs do not establish
N/M robustness for the final manuscript setup.

Verified archive:
`dildata:/data1/heejoonorm/OptiQ/studies/20260923_forward_sizes/campaign/source/experiments/kl_forward_size_1d/PROTOCOL.md`.
Its submission records source commit
`e7de5bc4416f5aa0cca0cab4a75cbf26e65019f2`.

The later high-L study uses the reported target family, N=M=128, B=32,
100K updates, and reverse L=1048576. Its stored configuration lists seeds
0–3. The manuscript states five seeds, so trace the submitted figure's
actual inputs before claiming which historical runs reproduce that figure.
A missing fifth seed must not be assumed or manufactured in provenance.

Follow-up audit: the archived 20260926_kl_paper_highL/EXECUTION.md and
PROTOCOL.md explicitly identify the main density figure as four seeds
(0–3), using the same two targets and high-L parent checkpoints. The new
sweep reports its own three seeds (0–2); no fifth historical seed is inferred.

Verified high-L archive:
`dildata:/data1/heejoonorm/OptiQ/studies/20260925_kl_six_highL`.
Original source commit: `41a020577dbf8a87d8f96d714ab3888a4bb82260`.
Vast execution source commit: `5c918efad2a78f797da6eb928fadc697a691de8c`.
COMPLETE records for all five relevant RKL targets, seeds 0–3, were found
across `campaign/runtime/confirm` and `vast_campaign/runtime/confirm`.
This establishes completion of those archived runs, not identity to every
submitted figure input.

## Controlled settings

| Parameter | Fixed setting |
|---|---|
| State / latent | Fixed state; scalar standard-normal latent |
| Action support | [-10,10] |
| Target value | Oracle Q(a)=0.25 log target_density(a), up to a constant |
| Critic learning | None |
| Architecture | Two hidden layers, 256 units each, GELU |
| Kernel | Normalized truncated Gaussian |
| Mean | 10 tanh(raw location) |
| Initializer | Location-head variance scale 3, fan_avg, uniform |
| Log standard deviation | Clipped to [-5,-1], initially -1 |
| Optimizer | Adam, learning rate 3e-4 |
| Training budget | 100,000 updates for every sweep run |
| Groups per update | B=32 independent groups |
| Forward objective | Original stopped-action, stopped-weight SNIS mixture NLL |
| Reverse objective | Original reparameterized marginal-score surrogate |
| Reverse score bank | L=2^20, independent of source-action latents |
| Bank sampling | Fresh per group/update; shared only across that group's M actions |
| Streaming | Original 4096-component chunks; unchanged numerical aggregation |

N and M are counts per group. L is a separate RKL approximation parameter
and is not part of this N/M sweep. In RKL, N controls the source latent bank;
L controls the marginal-score approximation. Do not reduce L for speed or
substitute the finite-N source-mixture score without declaring a new study.

### Verified meanings of N, M, and L

The archived `kl_mode_missing_search_1d/core.py` used by the high-L adapter
was read directly from dildata. `group_loss` draws z with shape (N,1),
draws M uniform component indices with replacement, and generates M actions.
In the RKL branch, `density_score` evaluates the independent L-bank at
those actions, and `loss=(a*direction).sum(-1).mean()` averages over M.
Thus M is used by both methods.

| Quantity | FKL | RKL |
|---|---|---|
| N | Source/proposal and student likelihood mixture size | Source latent bank size |
| M | SNIS candidate action count | Policy-action gradient Monte Carlo count |
| L | Not used | Independent marginal-density/action-score bank size |

At fixed parameters, each RKL action is marginally sampled from the full
semi-implicit policy after averaging over the fresh source latent bank.
Changing N changes source-latent reuse, correlations, gradient variance,
and cost; it does not replace L as a score-accuracy parameter. M changes
the number of action-gradient samples averaged. With independent source
and score banks, the mean RKL gradient estimator at fixed L does not depend
on N or M, whereas its variance and optimization trajectory do. Finite L
can still bias the score/gradient estimate.

RKL could instead sample a fresh latent for each action and omit a separate
source N. That is a different sampling design from the manuscript and is
not silently substituted into this controlled sweep.

Pair initial parameters across objectives and all N/M settings for each
seed. Verify hashes. Changing array shapes changes subsequent random draws;
paired initialization does not imply identical action trajectories.

## Three-seed sensitivity sweep and equal-trial tuning

Use tuning seeds 0,1,2 for both methods on every target. All runs complete
100K updates; do not prune runs using early mode coverage or early TV.

Use the full Cartesian product

`N in {32,128,512}` x `M in {32,128,512}`:

```
(32,32)    (32,128)   (32,512)
(128,32)   (128,128)  (128,512)
(512,32)   (512,128)  (512,512)
```

This is a regular logarithmic grid: each axis spans one quarter to four
times the original 128, in factor-four increments. All nine combinations
are measured, with no holes. It is a coarse sweep, not a full 5x5 sweep
over {32,64,128,256,512}; an intermediate optimum at 64 or 256 remains
possible. Describe any selected optimum as best within the tested grid.

The design separately measures N and M sensitivity around 128 and includes
their interactions at the smaller/larger extremes.

Nominal run count: 2 targets x 2 objectives x 9 settings x 3 seeds = 108.

For each target/objective, select the setting with the lowest mean final
per-seed histogram TV on the three tuning seeds. Record all grid outcomes.
Use W1 as a secondary measure; no selection by attractive plots or by the
FKL–RKL performance gap. Exact TV ties are resolved by lower mean W1, then
smaller N*M, then lexicographic (N,M) order.

Existing (128,128) runs may supply at most 12 sweep entries after verifying
their target, code/configuration, initialization, update count, and full
checkpoint provenance. Reevaluate them with the same final evaluation used
for the new sweep. Keep their original training commit distinct from the
new evaluation commit. Do not use old hardware timings as matched timings
for a new GPU pool.

## Run count and reporting without fresh-seed confirmation

There are 36 target/objective/configuration combinations and three seeds
per combination, for 108 total study entries. With all 12 eligible baseline
entries verified and reused, 96 new full training runs are needed. Until
reuse is verified, plan for 96–108 new full runs. Pilots, evaluation jobs,
and retries are not additional full-run settings but their GPU cost counts.

Do not launch the previously proposed seeds 10–14 confirmation stage.

Present default FKL vs default RKL, tuned FKL vs tuned RKL, and default FKL
vs tuned RKL. "Tuned" means best within the declared N/M search according
to the declared recovery metric, not a global optimum or all-hyperparameter
optimization. Selection and reporting use the same three seeds, so any
selected-best summary is descriptive and may be optimistic from selection.
Show all nine settings rather than claiming independent validation of the
selected setting.

## Unlimited-compute comparison design (discussion only)

For the N/M question under the manuscript implementations, the natural
design is a full factorial grid, at least
`{32,64,128,256,512} x {32,64,128,256,512}`, for both methods and targets,
with more paired seeds and separate confirmation seeds. Extend both axis
ranges in powers of two if needed to locate low-sample degradation and
high-sample saturation. Do not restrict N=M, since that cannot identify
separate effects or interactions.

A stronger study of whether RKL itself is adequately tuned would also
vary its independent score-bank L, and possibly optimizer settings, in
separate experiments. A method-native RKL experiment with fresh latent per
action would naturally use M and L as its two sampling axes. These are
distinct questions from the current matched-implementation N/M sweep.
No expanded-grid, L-sweep, or changed-sampler jobs are part of this plan.

## Evaluation and outputs

- Save learning snapshots at 0, 100, 1K, 10K, 25K, 50K, 75K, and 100K updates.
- At 100K draw 2^20 actual policy actions per seed using an independent
  evaluation RNG. Use the manuscript's 512-bin unsmoothed histograms.
- Primary metric: mean of per-seed histogram TV against exact target bin
  masses. Do not substitute TV of the seed-averaged histogram.
- Secondary metrics: 1D W1, individual target-region probabilities, and
  recovery rates under fixed archived region/coverage definitions.
- Keep target-region definitions and thresholds fixed before new results.
  Coverage does not by itself establish accurate within-region shape.
- Report all three individual seeds and mean +/- sample SD for every setting.
- Show both main-paper targets. Use common metric color scales across
  FKL/RKL 3x3 heatmaps. Do not interpolate unmeasured intermediate N/M values.
- Show N-only and M-only curves through (128,128), the 3x3 interaction grid,
  and default/selected-setting density panels and metric tables. Label
  selected-setting results as selected on these same three seeds, not held-out.
- Record synchronized training time, excluding compilation and evaluation,
  as well as end-to-end time and peak GPU memory. Plot recovery vs updates
  and training time. Main timing comparisons use matched hardware.
- The reverse surrogate's scalar value is not reverse KL. If objective
  diagnostics are reported, evaluate the marginal-density KL independently
  with a documented approximation; do not compare raw FKL/RKL training losses
  as a shared quality measure.

## Compute and execution preparation

The archived RTX 5090 validation measured about 0.234–0.236 seconds per
update at N=M=128, B=32, L=2^20: approximately 6.5 GPU-hours for 100K
updates, excluding evaluation and scheduling. Another archived GPU pool
measured roughly 0.87–0.89 seconds/update. These are historical timings,
not a current completion-time promise.

The sweep contains 54 RKL entries, so treating every entry as the baseline
speed gives about 351 GPU-hours before baseline reuse and FKL. This is a
baseline-equivalent illustration, not an estimate or lower bound for the
actual varying-N/M sweep.
Actual cost changes with M, hardware, and chunk execution; profile the
baseline and extreme shapes after compilation to estimate the real total.
Do not silently lower L or shorten RKL training to meet a guessed budget.

Before any new optimizer/GPU trial:

1. Resolve the submitted-figure input provenance and the four/five-seed
   discrepancy. Validate reuse candidates independently.
2. Prepare an isolated/reusable clean checkout from the relevant heejoon
   source while preserving existing unrelated untracked work.
3. Add a sweep launcher around the archived numerical kernels, explicit
   target allowlist, configs, evaluation, and this protocol.
4. Commit exact source/config/launch/protocol to Yonsei-DILLab/OptiQ branch
   heejoon. Record the full source SHA and file manifest before launch.
5. Verify unchanged baseline gradients, independent RKL score bank,
   correct N/M meanings, paired initialization, checkpoint restore, and
   evaluation-RNG isolation. Run small full-shape correctness/timing pilots.
6. Submit all eligible sweep jobs independently; do not gate different
   targets on each other's completion. Preserve preempted run checkpoints.

Store full checkpoints, samples, logs, manifests, and run/job IDs under
`dildata:/data1/heejoonorm/OptiQ/studies/20261107_kl_nm_sweep`.
Credentials and large generated artifacts remain outside Git.

## Required GPU-budget accounting

GPU accounting is part of the launcher, worker, checkpoint/resume protocol,
and report implementation. No production run may launch without it.

### Per allocation / attempt

Write an append-only attempt record with logical run ID, target, objective,
N/M/L/B, seed, phase (pilot/tuning/evaluation), source SHA,
configuration hash, scheduler job/array ID, hostname, GPU model and UUID,
GPU count and VRAM, software versions, UTC allocation/process start and end
times, exit status, final update, and resume lineage. Runtime durations use
a monotonic clock. Use scheduler accounting to reconcile killed attempts
that cannot write a final record.

Default to one independent run per GPU for interpretable accounting. If
sharing is introduced, record the allocation share and prevent charging a
whole GPU-hour separately to every process on the same GPU. Report
unattributable shared-device quantities at allocation level.

### Separate measured quantities

1. Allocated GPU-hours = sum over allocation segments of
   `GPU_count * allocated_seconds / 3600`. Queue time is excluded; compilation,
   evaluation, checkpoint I/O, and idle time inside an allocation are included.
2. Synchronized optimizer-training GPU-hours =
   `GPU_count * optimizer_training_seconds / 3600`, measured around JAX
   blocks with explicit device synchronization. Separately accumulate
   compilation/warmup, evaluation, and checkpoint/I/O durations; compilation
   warmup updates must be labelled and not counted twice.
3. Elapsed wall time and queue time, reported separately from GPU-hours.
4. GPU utilization, device memory usage, and power sampled every 5 seconds
   using NVML or nvidia-smi when available. Record sampling interval and gaps.
   Prefer a reset-aware cumulative device-energy counter; otherwise integrate
   measured power with timestamps to estimate kWh. Label estimated energy and
   missing telemetry explicitly. GPU utilization is not a substitute for
   allocated GPU-hours; device memory is not allocator-only memory.

Maintain `compute_attempts.jsonl`, a per-logical-run `compute_summary.json`,
and a campaign `gpu_budget.csv` grouped by target, method, N/M, and GPU model.
Timing/telemetry must not advance the training/evaluation RNG. Preserve
attempt metadata and accumulated counters across resume.

### Total campaign cost and comparisons

- Include pilots, correctness checks on GPUs, evaluation, failed/OOM runs,
  aborted runs, and retries in the total newly consumed campaign budget.
  Separately report successful production-training cost.
- Sum nonoverlapping allocation segments across resumes; never sum an
  already-cumulative counter a second time. Repeated work after rollback
  is genuine expenditure and remains in the total.
- Reused historical checkpoints have no new training charge. Record their
  historical training expenditure separately if available; new reevaluation
  is charged to this campaign. Do not invent missing historical timings.
- Report RKL and FKL tuning costs separately; no confirmation-stage cost is
  planned. Equal trial counts are not equal GPU budgets. Show quality vs updates and measured
  training GPU-hours, plus the total hyperparameter-search expenditure.
- Keep raw GPU-hours separated by hardware model. Do not claim an A6000-hour
  and a 5090-hour have equal computational throughput. Main timing comparisons
  use matched hardware. Report any normalized compute estimate separately
  with its benchmark and assumptions.
- Make a projected budget from measured pilots before full submission,
  reconcile it with accumulated actual cost during execution, and report both
  after completion. Do not infer a user-approved spending cap from an estimate.

## Interpretation commitments

Report whether RKL benefits from different N/M and by how much, even if the
original coverage gap shrinks or reverses. Do not retrospectively call 128
optimal based on unrelated earlier sweeps. This experiment tests sensitivity
on the two main-paper targets; it does not settle the reviewer's
separate questions about GMM40 baseline tuning or online-control baselines.

## Authorized execution details

- Compute host: heejoonorm@144.34.174.130:40326, four RTX 4090 24GB GPUs.
- Runtime root: /home/heejoonorm/OptiQ/kl_nm_20261107.
- Canonical archive: dildata:/data1/heejoonorm/OptiQ/studies/20261107_kl_nm_sweep/campaign.
- Dynamic independent queue: 96 new training entries, 12 baseline re-evaluations;
  no new five-seed confirmation. All full training runs use 100K updates.
- Baseline audit validates all numerical and target settings, source/parent
  commits, completion step, checkpoint hashes, and paired initialization.
  GPU preflight additionally restores each of the 12 checkpoints and evaluates
  1M actions with independent evaluation RNG before marking reuse valid.
- Twelve timing/correctness pilots: both targets and both methods at
  (32,32), (128,128), (512,512), seed 0, at true RKL L=1048576. Each compiles
  the original block shape, advances four blocks, then checks checkpoint
  resume across one extra block twice. Saved contents and integer/RNG states
  must be exact; subsequent floating-point leaves must agree at rtol=1e-5,
  atol=1e-6. All pilot GPU time is accounted for.
- Pilot source arrays cover baseline and extreme shapes; the full grid still
  consists of nine pairs and runs irrespective of pilot quality metrics.
- GPU training gate requires all 12 pilots and all 12 baseline restorations/
  re-evaluations to pass. It is a correctness gate, not a target-completion gate.
- Pinned environment: JAX/jaxlib 0.5.3, Flax 0.10.4, Optax 0.2.4,
  NumPy 2.2.6, SciPy 1.15.3. Save the full installed package lock outside Git.
- JAX_THREEFRY_PARTITIONABLE=false preserves the archived initialization;
  XLA_FLAGS uses library GEMM and WHILE_LOOP_UNROLLING_NO_UNROLL as archived.
- Archived numerical kernels are copied byte-for-byte. FKL uses the archived
  100-update blocks and RKL five-update blocks; training, evaluation, and
  atomic checkpoint writing are instrumented by a separate wrapper.
- Four supervisor workers, one exclusive GPU each, with full optimizer/RNG
  checkpointing and GPU locks. Source manifest and full commit are checked
  before execution and on resume. Failed attempts remain in the cost ledger.
- Device telemetry is sampled about every five seconds. Pilot and production
  reservations are separate; worker reservation hours include between-task
  overhead and must not be added to their overlapping per-attempt hours.
  This is compute accounting, not an inferred cloud dollar invoice.
- Per-GPU append-only compute_attempts_gpu*.jsonl streams avoid concurrent
  writers; monitor.py aggregates them into per-run summaries, gpu_budget.csv,
  and CAMPAIGN_STATUS.json every minute. Attempts keep BEGIN/heartbeat/END
  files so interrupted attempts remain identifiable.
- dildata pulls the campaign every 180 seconds over an existing backup key
  whose public key is restricted on this new host to read-only rsync of this
  campaign directory. No private key transfer or general shell access is used.

### Preflight correction before production

Initial preflight source 0bf53e40810ceef15b9cfe922e1b066849b91151 used an
overly strict byte-identical continuation check. Diagnostic source
c5155f6f951a291cbb1468efdb6bfe59fd2cbc27 verified exact checkpoint contents
and RNG, while repeating the same GPU-resident RKL state without any restore
also varied at float32 precision (largest observed optimizer-state difference
about 1.05e-7; parameter difference below 7.5e-9). Thus continuation uses
the explicit tolerance above; the archived numerical kernels are unchanged.
Failed pilots are not automatically retried. An explicitly resumed pilot
starts fresh, and its bounded step loop cannot overrun its pilot budget.

Keep the original source/runtime immutable. Corrected execution lives in
runtime roots revisions/v2 and revisions/v3 under the original campaign root. HISTORY_ROOTS.json
includes the original campaign's attempts and reservations in all new budget
reports, including failed, paused, and diagnostic GPU work. The original
read-only backup root covers both revisions. No production training used the
initial validation wrapper.

The v2 check found a 2.8313e-7 maximum difference in one Adam first-moment
leaf at the 20-update Gaussian pilot; checkpoint contents, RNG, and integer
states remained exact. Final floating-point continuation tolerances are
rtol=1e-5, atol=1e-6; this changes validation only, not training arithmetic.
Preserve successful v2 pilot and baseline-evaluation records with their
original source commit, after verifying identical archived kernels and
configuration hashes. Their tighter passed tolerance implies the final
tolerance. Retry only the failed pilot from fresh initialization in v3.
VALIDATION_INPUTS.json records prior provenance explicitly; no old result is
relabeled as generated by the new commit. Include both prior roots in cost
aggregation and retain their full samples/checkpoints at those roots.
