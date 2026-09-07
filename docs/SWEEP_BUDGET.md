# 432-run sweep: runtime, 72-hour capacity and Vast budget

Date: 2026-09-07, approximately 13:00 KST. **Planning estimates, not a purchase
order or a completed benchmark of the optimized branch.** No rentals have been
created; existing experiments have not been stopped or changed.

## 1. Scope

36 configurations × 4 tasks × 3 seeds = **432 runs**, each 1M environment steps.
One experiment per RTX 4090. Common settings and axes are in
[PARAMETER_SWEEP.md](PARAMETER_SWEEP.md). The 72-hour clock below starts when
the sweep fleet is provisioned, not while waiting for existing runs to finish.

## 2. Measurements vs assumptions

The [runtime evidence snapshot](sweep_runtime_evidence_20260907.json) records the
completed-run wall times, Humanoid timing window and checkpoint size.

| Evidence | Time | Interpretation |
| --- | ---: | --- |
| 12 completed Dog runs on vast2 | 10.61–11.67 h; median 11.18 h | Actual start/end wall times; old every-step diagnostics and 5K evaluation |
| Completed Dog run, beta 0.1, seed 1 | 10.79 h | Actual 1M completion, not extrapolation |
| Current DMC Humanoid run, 95K→150K | 1,846.861 s / 55K | 9.33 h/1M extrapolation; not completed, old settings |
| New sparse-diagnostics / 10K-eval branch | GPU benchmark pending | CPU smoke tests are NOT GPU timing evidence |
| MuJoCo Humanoid-v4 and MyoSuite | No matched timing measurement | Do not label any task estimate below as measured |

The optimized branch reduces diagnostics/host transfers and halves scheduled
evaluation calls, but does not shrink the large critic or candidate batch.
Do not assume evaluation halving means training time halving. Different sigma,
temperature and fixed beta values have the same tensor shapes; anchor removal
reduces 80 candidates to 64, but does not reduce the entire run cost by 20%.

Provisional per-task planning bands, spanning BOTH anchor settings:

| Task | Runs | Provisional h/run | Provisional total GPU-hours |
| --- | ---: | ---: | ---: |
| MuJoCo Humanoid-v4 | 108 | 8–12 | 864–1,296 |
| DMC Dog run | 108 | 8–11 | 864–1,188 |
| DMC Humanoid run | 108 | 7–10 | 756–1,080 |
| MyoSuite pen-twirl-hard | 108 | 8–16 | 864–1,728 |
| Total | 432 | Weighted mean 7.75–12.25 | 3,348–5,292 |

These are explicit planning assumptions, not confidence intervals. MyoSuite
compatibility and return/value support remain unresolved; setup time could
exceed the allowance. No purchase commitment should depend on that row yet.

## 3. GPUs needed for a 72-hour target

Capacity model: 4 hours for provisioning/validation/aggregation; 15% time
allowance per run; complete-run batching rather than fractional jobs.

```
jobs_per_GPU = floor((72 - 4) / (mean_run_hours * 1.15))
required_GPUs = ceil(432 / jobs_per_GPU), rounded up to a multiple of 8
```

| Mean h/run scenario | Work (GPU-h) | Jobs/GPU in window | Planned 4090 count | 8 GPUs, ideal days | GPU-only 72h reservation at $0.40/GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 h: optimistic | 3,456 | 7 | **64** | 18.0 | $1,843 |
| 10 h: central | 4,320 | 5 | **88** | 22.5 | $2,534 |
| 12 h: conservative | 5,184 | 4 | **112** | 27.0 | $3,226 |

This is a homogeneous-duration planning model, not a deadline guarantee for
heterogeneous jobs. Once timings exist, use longest-estimated-job-first scheduling
and a task-specific simulation; start long MyoSuite/Dog jobs early. For example,
a task taking 16 hours cannot be packed as five 10-hour jobs on one GPU.
Fleet-wide mean and total GPU-hours alone do not guarantee the final straggler
finishes within 72 hours.

Currently 8 GPUs are rented but only **7 count as healthy**: vast1 GPU 1 has
repeated historical CUDA errors and is excluded. All seven are busy. At 10 h/run,
seven available-from-start GPUs would take approximately 25.8 ideal days for
432 runs; current jobs add waiting time. If all seven later join a sweep, an
88-healthy-GPU target still needs **81 additional healthy GPUs** (round up to
the rentable node sizes). The 8-GPU costs above are full-fleet examples, not
an instruction to replace or stop the current instances.

## 4. Current Vast quotes and budget

Read-only CLI query: RTX 4090, at least 4 GPUs, rentable/unrented verified hosts,
reliability >0.99, on-demand, at least 300GB disk, price calculated for 300GB
allocated storage per instance. Same-machine alternative offers were deduplicated;
offers with less than 4 days remaining rental duration were excluded.

See [raw snapshot](sweep_market_snapshot_20260907.json) for offer IDs, rates,
CPU/RAM, storage and bandwidth. Availability is **not reserved**. Host reliability
scores are not an uptime SLA, and a short pilot is required before trusting a host.

- Low-priced examples: 8 GPUs + 300GB disk at **$2.57/h**, another at **$2.75/h**;
  4 GPUs + 300GB at **$1.42/h**. These are node prices, not per-GPU prices.
- Query returned 24 offers. Deduplicated eligible capacity in this restricted
  query was **99 GPUs**; that does not mean all global inventory is only 99.
- A greedy, unique-machine example reached **89 GPUs** (includes a 9-GPU offer)
  at **$37.522/h including allocated storage**, or **$2,701.58 / 72h**.
- This covers the central 88-GPU planning target. The 112-GPU conservative
  target was **not covered by this particular filtered snapshot**.

| Budget item, central 89-GPU quote | USD |
| --- | ---: |
| 72h compute + 300GB allocated disk per selected node | 2,702 |
| Output backup/network allowance | 50 |
| Additional cash contingency, about 10% | 275 |
| Recommended rounded budget | **3,100** |

For higher market rates or a 12-hour average, hold **$3,800–4,500** pending a
new availability quote. These are total new-fleet equivalent budgets. Existing
instances' actual billing must be added/substituted if retained; it was not
queried here. Tax, card/FX fees and external long-term archive storage are excluded.
At an explicitly assumed budgeting exchange rate of KRW 1,500/USD (NOT a live FX
quote), $3,100 ≈ KRW 4.65 million and $4,500 ≈ KRW 6.75 million.

Compute-only sensitivity for 88 GPUs held for all 72 hours:

| Rate / GPU-hour | 72h GPU cost |
| --- | ---: |
| $0.30 | $1,901 |
| $0.40 | $2,534 |
| $0.50 | $3,168 |
| $0.60 | $3,802 |

Compute can be billed for less than 72 hours if nodes are released after their
last job and backups, but no instance will be stopped/destroyed automatically
without appropriate user authorization. A faster fleet mainly buys elapsed time;
it does not remove the sweep's GPU-hour requirement.

### Storage and interruption costs

Measured completed Dog checkpoint directory: **3.38GB/run** with 50K checkpoints.
432 similar runs would produce **about 1.46TB** of checkpoints, before logs and
environment differences. Provision approximately 300GB per 8-GPU node and verify
free disk; archive incrementally. Do not assume a 100GB instance can retain a
whole multi-wave queue. The quoted 300GB/node storage is already included in the
$2,702 figure, so do not charge it a second time.

Current actor/critic checkpoints do not contain complete replay/environment/RNG
resume state. Therefore **on-demand, not interruptible/spot**, is the planning
default: cheap interrupted jobs may require expensive restarts. No hardware
or instance type guarantees freedom from crashes.

Vast's primary documentation confirms separate compute, persistent storage and
bandwidth charges, and that stopping compute does not stop storage billing:
[Vast pricing documentation](https://docs.vast.ai/guides/instances/pricing).

## 5. Lower-budget alternative (changes the original experimental design)

Do not silently execute this instead of the full matrix:

1. Screen 36 combinations on all 4 tasks using seed 0: **144 full 1M runs**.
2. Select four common configurations using a predeclared cross-task criterion;
   add seeds 1 and 2 on all 4 tasks: **32 runs**.
3. Total **176 runs**, with three-seed results for the selected configurations;
   unselected configurations have only one seed, not three.

At 10 h/run, 40 GPUs allow four screening waves + one confirmation wave:
`4h setup + (4+1)*10h*1.15 = 61.5h`, assuming homogeneous durations and rapid
selection. Approximate GPU-only reserved cost at $0.40 is $1,152 for 72h;
with storage, backup and contingency, budget **$1,300–1,600**.
This is not equivalent evidence to the full 432-run three-seed factorial sweep.

## 6. Pending timing probe

`scripts/benchmark_sweep.py` is a bounded offline timing script. It acquires the
same per-GPU file lock as the existing queues and also checks NVIDIA active PIDs.
It refuses vast1 GPU 1, never kills a process, and does not log to W&B.

On the next healthy idle GPU, measure Gym Humanoid, Dog run and DMC Humanoid
with/without anchors (6 probes), plus Dog with every-step diagnostics for an
A/B comparison (one additional probe). Only include MyoSuite after environment
compatibility is confirmed without upgrading the running shared environment.
Each probe uses full-sized networks and batch/UTD, 5K replay warm-up, 1.1K burn-in
to compile both diagnostic paths, 1K timed updates, ten stochastic evaluation
episodes and one checkpoint save. Report per-task timing as provisional until
a longer segment or complete run confirms it. Sigma/T/beta need not each be
benchmarked separately because their tensor shapes are unchanged.

The script passed a CPU-only, tiny-network Pendulum smoke test. Its CPU timing
is explicitly marked and **must not be used in these GPU budget estimates**.

**Scheduling status:** the requested follow-up heartbeat was rejected by the
permission reviewer and was NOT registered. GPU benchmarking remains pending;
explicit permission for recurring remote checks/automatic bounded probes is
needed, or the user can request a one-time check when a GPU becomes available.
