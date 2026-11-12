# Pure online Adroit Sparse-v2: SAC, TD3, and iBOLT

User authorization, 2026-11-12: run seeds **4 and 5**, vary only vector environment
count and UTD, start with 256 environments, use four GPUs on the supplied server.
The user then expanded the allocation to **eight GPUs across two servers**.
Use branch **heejoon-rebuttal**. This supersedes the earlier seed5/6 pilot and the
workspace's older branch instruction for this study only. The initial campaign
excluded iBOLT; the later authorization below adds one fixed-temperature iBOLT
comparison. No alpha/noise tuning or automatic confirmation on seeds0-3.

Reference: heechan-rebuttal ebdc46e59db1642902e0331c1e3aa662556c6553.
The existing A1 demonstration-based experiment is not resumed or modified.
The active servers are heejoonorm@175.155.64.205:19159 (shard0) and
heejoonorm@179.255.127.236:31020 (shard1). Both expose four RTX5090 GPUs.
Each server receives all four tasks and both algorithms; shard0 has SAC seed4
and TD3 seed5, shard1 has SAC seed5 and TD3 seed4. All profiles keep this mapping.
The originally supplied
203.75.167.56:42913 failed device access and has no training jobs from this study.

## Fixed scientific settings

- Door, Hammer, Pen, Relocate: native `AdroitHand*Sparse-v2`, horizon200,
  upstream Gymnasium-Robotics b8da02a89191daf8f004e7f7510ff655f7a90100.
  Reward is +10 on native success and -0.1 otherwise. Preserve resets/physics.
- Randomly initialized actor/critics and empty standard online replay only.
  No demonstrations, checkpoints, BC, HER, shaping, curricula or normalization.
- SAC: unmodified shared `baselines/jax_sac.py:JaxSAC`, actual JAX GPU backend,
  auto-alpha and target entropy minus action dimension. SB3 reference initialization.
- TD3: unmodified SB3 2.7.0 learner, exploration Gaussian std0.1 in [-1,1]
  coordinates, policy delay2, target noise0.2 clipped at0.5; deterministic evaluation.
- Both: actor/critic256x2 ReLU, twin critics, LayerNorm off, lr3e-4,
  gamma0.99, tau0.005, minibatch256, replay1M, four CPU collection workers/run.
- Requested3M interactions. Use **3,000,064** for every profile (next multiple
  of256, excess64=0.00213%). Warmup **10,240** (10k rounded to256) is identical
  across profiles. Evaluations do not count toward training interactions.
- Only profiles differ: (env256,UTD0.25), (env256,UTD1), (env64,UTD1).
  Each has2methods x4tasks x2seeds=16 runs;48 full runs in total.
  Priority order follows this list, but independent tasks have no afterok gates.
- Collect one vector step, then n_envs*UTD critic updates. Twin-Q optimization
  counts as one update; TD3 actor updates every other critic update.
  Expected critic updates:747,456 or2,989,824.

## Evaluation and provenance

- Initial policy and a separate uniformly random-action diagnostic, each100
  episodes. Regular evaluations every50,176 transitions use100 episodes;
  final evaluation at3,000,064 uses1000 episodes. SAC deterministic primary,
  stochastic secondary; TD3 deterministic primary. Fixed reset schedules are
  paired by training seed across algorithms/profiles and milestones.
- Evaluate after the update block. Evaluation preserves Python/NumPy/Torch/JAX
  learning RNG and never writes replay. Report any-step native success, final-step
  success, success step count and >=25 success steps (not necessarily consecutive).
- Record first online success as vector-step-end transition count and environment
  index, warmup successes, all episode success counts and wall times. Initial/random
  successes and isolated training successes do not establish learned competence.
- CPU simulator checks verify rewards, shapes, seed determinism and native timeouts.
  GPU preflights verify real learner updates, timeout bootstrapping, evaluation RNG,
  finite parameters and exact update counts before full runs. Smoke results are
  separate, not part of the benchmark. No CPU fallback for reported learning.
- Commit source/configuration/protocol/launch scripts before smoke or full runs.
  Deploy an immutable hash manifest with full source SHA, branch, runtime versions,
  run IDs, GPU allocation and process IDs. Do not claim bit-exact resume: saved
  policies, optimizers and final replay support analysis but environment/RNG state
  is not fully resumed by this launcher. Failed runs retain outputs, never overwrite.
- Use Supervisor for the controller, with a GPU memory admission guard and
  staggered compiler starts. Start at up to2 jobs/GPU if memory/CPU checks permit;
  adjust operational concurrency from measured throughput without scientific edits.
- Store results/checkpoints outside Git; archive to
  dildata:/data1/heejoonorm/OptiQ/runs/adroit-online-20261112.
  Retain inference checkpoints every250,880 and final; final replay is saved.

## Execution

The deployment script verifies clean tracked inputs, creates a git-archive source
snapshot and manifest, and transfers it to the explicitly supplied server. Runtime
installation reuses the pinned Adroit requirements. The controller requires an
explicit GPU-preflight receipt tied to the source SHA; it refuses unverified starts.
Never start jobs from mutable working copies. The monitor produces JSON/CSV summaries
with current steps, first success, evaluation success and throughput per run.

## iBOLT extension — user authorization, 2026-11-12

The user requested "same current settings, iBOLT temp=0.25, dilg-01 GPUs0/1".
Run **eight** fresh iBOLT models: all four tasks x seeds4/5, env256/UTD0.25,
3,000,064 transitions (747,456 optimizer updates), warmup10,240. No other iBOLT
temperature/profile/seed sweep is included. Existing SAC/TD3 immutable source
e1a244a8216d3286fee1dbcc84fd3adfb748dabb and running jobs remain unchanged.

Use the same native robotics source, horizon200, sparse reward, grouped four-worker
collector, empty standard replay1M, batch256, width/depth256x2, lr3e-4, gamma0.99,
tau0.005, evaluation reset/action-seed schedules, 100 regular/1000 final episodes,
and evaluation/checkpoint step schedule. No demonstrations, BC, HER, observation
or reward normalization, LayerNorm, BatchRenorm, or CrossQ.

The repository's `exploration.ExplorationIBOLT`, `ibolt/` and `models/critic.py`
numerical source is reused unchanged. Algorithm-specific defaults are explicit:
GELU actor/critic (SAC/TD3 use ReLU), N=M=64, density correction beta1, mean twin-Q
proposal scoring, ordinary min twin-Q TD backup without SAC entropy bonus, actor
update every critic update, conditional log sigma[-5,-1] initialized at-1.
Actor temperature is fixed0.25. DACER behavior exploration remains on, lambda0.1,
initial alpha0.27, alpha lr0.03, target entropy -0.9 per action dimension, update
interval10,000 learner updates, 200 samples and3 mixture components. Its adaptive
alpha is intrinsic to the existing algorithm; it is not a new hyperparameter sweep.

Primary iBOLT evaluation uses a fresh random latent with conditional mean
(`stochastic_z`), consistent with the current repository Adroit comparison.
Also save zero-latent mean (`zero_z`) and full mixture (`stochastic_z_sigma`),
all without external DACER noise. Never label stochastic-z as a deterministic
policy. All modes use the same reset episodes as SAC/TD3 and record any-success,
success-step count, >=25 success steps (not consecutive), and final-step success.
Evaluation preserves learner, policy, DACER, NumPy, Python, and Torch RNG streams.

Use only dilg-01 GPUs0/1 (RTX PRO6000 Blackwell Max-Q,96GB each), initially two
concurrent jobs/GPU with eight logical CPUs/job, including the same four collection
workers. Four tasks and both seeds appear in the first wave; remaining jobs are
independently eligible with no cross-task completion dependency. Do not interrupt
other jobs or use GPUs2/3. Protect starts with GPU PID/memory checks and a controller
lock. Run under a dedicated tmux session; no automatic retry/overwrite/resume.

Local runtime: `/home/heejoonorm/adroit-online-ibolt-20261112/venv`, Python3.11.17
and the exact installed package lock from baseline shard0. Sources, manifests,
runtime receipts, evaluations, logs and checkpoints are written directly to
`/mnt/data1/heejoonorm/OptiQ/runs/adroit-online-ibolt-20261112`, the NFS mount of
`dildata:/data1/heejoonorm/OptiQ/runs/adroit-online-ibolt-20261112`.
Commit before any smoke/full learning; deploy a fresh immutable manifest.

Preflight uses existing distribution/config checks and a real native-task smoke
on each of the four tasks. Require actual JAX GPU arithmetic, seeded trajectory
agreement, native reward/horizon checks, empty standard replay, evaluation RNG
preservation, finite updates, exact optimizer counts, timeout bootstrapping and
exact actor/critic optimizer checkpoint restoration. The shared runner saves
native Flax checkpoints, separate DACER/RNG state and final replay. Simulator
state is not saved; these files do not imply a bit-exact interrupted-run resume.

## Additional iBOLT temperatures — user authorization, 2026-11-12

The user requested additional temperature0.05 and0.10 experiments. Add16 fresh
runs: each temperature x Door/Hammer/Pen/Relocate x seeds4/5. Match T0.25 exactly
except actor temperature and run/output identifiers: env256, UTD0.25,3,000,064
transitions,747,456 updates, warmup10,240, all native sparse/environment/replay/
network/DACER/evaluation settings above. Preserve the same paired evaluation
seeds. Existing T0.25 run IDs and all24 iBOLT runs are distinct. No warm-start,
result-based early stopping, automatic temperature choice, or seed0-3 follow-up.

The original T0.25 immutable source86354642a0936688d3b268f1c8ba10e74d1480b0,
controller, running PIDs, queue and outputs remain unchanged. The new queue is
`/mnt/data1/heejoonorm/OptiQ/runs/adroit-online-ibolt-temps-20261112`, directly on
the same dildata NFS storage. Reuse the exact private116-package runtime. Commit
the extended configuration, protocol and launch scripts before GPU smoke/full
learning; write the new full SHA and manifest beside every new run.

Run only on dilg-01 GPUs0/1. The new controller permits at most2 additional jobs
per GPU, at most4 total including existing T0.25 jobs. CPU affinity is operational
and may overlap the existing jobs; preserve4 collection workers and8 logical
CPUs per learner. Current measured CPU load is under5 on32 logical CPUs and GPU
memory use under3GB of96GB per device, permitting this additional concurrency.

The peer-aware admission guard recognizes only the explicitly authorized T0.25
queue, verifies live process ownership, command/output, immutable source cwd,
spec source SHA, temperature and GPU assignment, and rejects unrelated GPU PIDs.
The old controller does not know the new queue, so wait until its already queued
seeds have STARTED before introducing new GPU work. Do not wait for their learning
to complete. This prevents starving its remaining starts without restarting or
editing it. New runs remain independently eligible; no task-completion gates.

Preflight checks each new temperature on each native task:8 smoke runs total,
up to4 concurrent (two/GPU), plus the existing config/distribution checks.
Require real GPU updates, finite metrics, exact counts, empty online-only replay,
unchanged evaluation RNG, timeout bootstrapping and checkpoint restoration.
Admission checks and preflight receipts name both temperatures. Each job passes
its temperature explicitly through CLI to the resolved actor config; a regression
check verifies temperature is the only scientific configuration difference.

## Move pending iBOLT jobs to RTX5090 — user authorization, 2026-11-12

The user requested freeing dilg-01 for another task and prioritizing iBOLT.
Move exactly the eight still-unstarted jobs recorded in `migration_5090.json`:
Door/Pen seed5 and Hammer/Relocate seed4, each T0.05/T0.10. No restart/resume of
the four already running dilg learners, and no numerical or budget change.
Split four jobs per existing5090 host, including every task and both temperatures
on each host. The retained run IDs are globally unique; origin output directories
must not exist. Record origin delegation before enabling destination controllers.

Send SIGTERM to each verified original controller PID only. The existing signal
handlers stop dispatch and drain their active children. Do not use Supervisor
stop/group signals. Save the command, owner, start ticks, source manifest and
queue snapshot. The old baseline controllers have autostart/autorestart disabled.
Record `queue-delegation.json` at the dilg root immediately; a metadata-only
finalizer marks its eight stale pending entries migrated after the old controller
exits. While draining, the delegation record overrides those pending entries.

The5090 `priority-plan.json` and `priority-queue-state.json` become authoritative
for each host's24 baseline jobs plus four migrated iBOLT jobs. Existing baseline
learners are adopted for monitoring/resource accounting; their processes and
source commit e1a244a8216d3286fee1dbcc84fd3adfb748dabb are preserved. Unstarted
baseline jobs use the same original snapshot and runtime after higher-priority
iBOLT dispatch. There is no environment-completion gate. No automatic retries,
overwrite of run folders, or result-based selection is allowed.

Resource budget: two baseline slots plus one iBOLT slot per5090, at most three
total learners per GPU. Each has8 logical CPUs and four collector workers. The
new iBOLT CPU blocks64-95 are disjoint from baseline blocks0-63. Both hosts have
more than120 effective cgroup CPU cores, over250GB available container memory,
over95GB free disk and over29GB free GPU memory at migration inspection. Do not
admit an unrelated GPU PID or a GPU with less than12GB free memory. Baseline
capacity remains two slots/GPU after iBOLT completion.

Reuse the116-package runtimes. Verify installed package versions and native
robotics assets across hosts; shard1's freeze differs only in local archive URLs
for the same pinned Torch wheel and Gymnasium-Robotics archive. Perform a real
GPU/native-task/learning/checkpoint smoke for each migrated task/temperature on
its target GPU before its full run. This preserves the prior53,248-step/512-update
smoke protocol. Record per-job receipts, old and new full source SHAs, unchanged
learner/config hashes, controller PID and learner PID/GPU UUID. Commit all new
scheduling inputs before smoke/full starts. Store new results under the existing
Vast root's `runs/` so central dildata archiving includes them automatically.

Archive termination must use the authoritative priority queue, including migrated
iBOLT jobs. Preserve original legacy queue files while their controllers live;
after they exit, mirror the24 baseline statuses from the priority queue for
compatibility. No keys or credentials are transferred as part of this migration.

## Hammer CUDA failure recovery — user authorization, 2026-11-12

The user explicitly requested diagnosing and rerunning Hammer seed4 T0.10.
The first5090 attempt aborted at1,003,520 transitions with SIGABRT and JAX/PJRT
CUDA_ERROR_ILLEGAL_ADDRESS. CUDA copy-stream errors do not identify the original
faulting kernel. No host/cgroup OOM was recorded; kernel Xid logs are unavailable
inside the container. Do not claim a proven compiler/driver defect without a repro.

Preserve the failed folder and all original controller/learner processes. Use one
explicit retry, `hammer-ibolt-s4-e256-u025-t010-retry1`, on shard0 GPU1's vacant
iBOLT slot, CPUs72-79. Its separate `recovery-queue-state.json` overrides the
logical run's failed status for reporting; retain the original queue as history.
Before the retry, load the1,003,520 checkpoint into separate diagnostic processes:
256 updates under Compute Sanitizer memcheck, then8,192 normal updates including
DACER, repeated exact checkpoint round trips, and native evaluation. Diagnostic
replay is newly collected native random transitions and is NOT a scientific resume.

Only after both diagnostic processes pass, start the full original scientific
configuration from scratch with seed4, fresh parameters, empty replay,3,000,064
transitions. Intermediate replay/simulator state was not saved, so continuing the
checkpoint would not preserve the intended experiment. No speculative changes to
temperature, exploration, architecture, loss, environment, shared packages or
drivers. A fresh CUDA process clears the aborted context; a successful retry alone
does not prove the root cause. Full Python tracebacks/faulthandler aid a recurrence.
No automatic second retry and no selection based on reward/success. Commit and
manifest diagnostics, launch script and protocol before running them. Archive both
attempts and recovery metadata to the same dildata campaign destination.

The initial diagnostic (b15bb764cdaaadc4a605c5d35070cc73c2171fed) crashed under
Compute Sanitizer during XLA GEMM fusion autotuning, repeatedly reporting
`Red zone modified`, before any full retry. This is a reproduced compiler-path
failure under instrumentation, not proof that it caused the original asynchronous
illegal-address error. Preserve that diagnostic's state/logs and test the narrow
process-local workaround `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`.
Compare one identical checkpoint/batch/RNG update on the original and fallback
GPU paths, including actor/critic/optimizer/target/RNG leaves (rtol1e-5, atol1e-6).
Require the fallback memcheck and8,192-update DACER/checkpoint stress to pass before
the fresh full run. No shared runtime mutation. Record the compiler flag with the
new run; different GPU kernels may introduce floating-point rounding differences.

The uninstrumented original probe also emitted GEMM `Red zone modified` messages;
the fallback did not. Both one-update probes completed. The initial all-leaf
atol1e-6 comparison rejected four elements; inspection of all53 leaves found max
absolute difference1.460314e-6, parameter relative-L2 at most1.286278e-6, optimizer
relative-L2 at most0.001096, and exact integer counters/RNG. Preserve those probes
and the failed strict comparison. For the reviewed cross-kernel float32 check use
atol5e-6/rtol1e-5 plus per-leaf relative-L2<0.002 and exact integer/RNG equality.
This is numerical agreement within tolerances, not bitwise identity. Only the
diagnostic controller is continued; a full scientific retry has not started yet.

The Triton-disabled memcheck then identified an actual out-of-bounds16-byte global
read in `loop_multiply_fusion_1`, launched through `CudaCommandBuffer::LaunchGraph`
and `CommandBufferThunk`. It exited86; no full retry started. Preserve that log and
queue state. Also disable XLA CUDA graph command buffers in this process:
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer=`.
Before memcheck/stress/full training, compare another one-update probe with the
previous cuBLAS-path probe, keeping exact RNG/counter checks. Neither compiler
flag changes the intended loss, samples, update count, replay or environment.

The graph-disabled one-update probe is bitwise identical to the cuBLAS-only probe
on all53 leaves. Nevertheless the old CUDA12.8 Compute Sanitizer2025.1 reports an
invalid read in another kernel (`loop_add_fusion`) with graphs disabled. Therefore
do NOT identify command buffers alone as the root cause. The ordinary uninstrumented
8,192-update stress with both flags passed, including DACER, native evaluation,
finite states and exact checkpoint round trips. Cross-check the memory findings
using NVIDIA's official standalone cuda-sanitizer-13-2=13.2.87-1, matching the host
driver's CUDA13.2 capability. Extract it under setup only, never install a system
driver/toolkit or alter training wheels. Package SHA256:
8899b0126b75be7ff0df6b11b685312b180c5928f567c67f1e4962d4df6b4aad.
Require that256-update memcheck to pass before full retry. Preserve all earlier
failed diagnostics; no suppressed sanitizer errors or ignored nonzero exits.

The current sanitizer reported zero memory errors but the256-update checkpoint
finite-state assertion failed; this is NOT a passing diagnostic. Next isolate
asynchronous dispatch/input lifetime by enabling the process-local
`ADROIT_JAX_SYNCHRONIZE_UPDATES=1` adapter. It retains the complete update argument
tree and blocks until each existing JIT update returns, without changing random
draws, sampling, optimizer expressions, UTD or environment collection. Require an
exact53-leaf one-update comparison against the previous path, a fresh256-update
CUDA13.2 memcheck including finite-state checks, and a fresh8,192-update stress.
Only then start the full from-scratch retry. This changes execution timing only;
it is isolated to the authorized process and does not mutate running snapshots.

Synchronization also failed the instrumented diagnostic. Do not treat it as a
verified fix or begin the full retry. Compare the same saved-checkpoint diagnostic
on another available5090 GPU, retaining failed evidence, to separate device-local
faults from software/runtime faults. Hardware-only exact integer arithmetic probes
may use available diagnostic slots on GPUs0/1 and compare JAX/Torch against NumPy.
These contain no RL learning and are not benchmark results. Keep all peers running.

Final recovery placement supersedes the GPU1/flag experiments above: GPU0 passed
CUDA13.2 memcheck with256 original-runtime updates, finite checkpoint round trip
and native20-episode evaluation (source857000e210b38d1cebac69f333636a8747e7479b).
Both devices passed the integer arithmetic probe. This does not establish a
hardware defect: instrumented GPU1 and GPU0 tests are not all identical, and a
passing short test cannot rule out an intermittent compiler/runtime/device fault.
Use shard0 GPU0, CPUs64-71, with original JAX flags (unset), synchronization
adapter OFF, and unchanged wheels/scientific settings. Extend original-runtime
GPU0 memcheck to4,096 updates and uninstrumented stress to8,192 updates, both
crossing the250,000-update DACER boundary. Require finite states, checkpoint
round trips and native evaluation. Then launch the single fresh3M retry under
Supervisor. Keep all failed diagnostics and report the original failure as a
CUDA illegal address with unresolved lower-level cause, operationally mitigated
by a fresh process on a validated different GPU. Do not describe discarded
Triton/graph/synchronization experiments as verified fixes.

## Temperature0.01 extension — user authorization, 2026-11-12

The user requested "temp0.01도 돌려봐". Add exactly eight fresh pure-online runs:
Door/Hammer/Pen/Relocate x seeds4/5, actor temperature0.01. All other scientific
settings remain identical to the previous iBOLT temperatures: env256, UTD0.25,
3,000,064 transitions, warmup10,240,747,456 updates, standard empty replay1M,
native Sparse-v2, DACER on, original learner/runtime, and paired evaluation seeds.
Keep the original evaluation modes,100 regular/1000 final episodes, and report
any-success and cumulative>=25 success steps separately. No checkpoint warm start,
demonstrations, altered exploration, additional temperatures, or seed0-3 selection.

Use `temp001-queue-state.json` under each existing5090 root, independent of the
priority and Hammer recovery queues. Seed4 goes to shard0 and seed5 to shard1.
Every task is independently eligible; Hammer/Relocate have scheduling priority,
not completion dependencies. Preserve the two baseline slots and one iBOLT slot
per GPU with disjoint8-CPU allocations. Shard0 GPU0 remains occupied by the
authorized Hammer retry until it finishes. Exclude shard0 GPU1 from new iBOLT
placement because its prior diagnostic failures remain unexplained; do not stop
its existing baselines or claim a proven hardware defect. Eligible GPUs are
shard0 {0,2,3}, shard1 {0,1,2,3}: initially six new runs can start, two wait.
Use original execution flags; do not enable abandoned recovery workarounds.

Before each full run, run the established native task/temperature smoke on the
same target GPU (53,248 transitions,512 updates, seed4 diagnostic). Check actual
GPU arithmetic, native physics/reward/timeouts, empty replay, preserved evaluation
RNG, finite parameters and exact checkpoint restoration. Smoke data are separate.
The full run uses its assigned seed and fresh parameters/replay. Controller PID,
full source SHA/manifest, runtime packages, GPU UUID, CPU affinity, smoke and full
PIDs and receipts are recorded. Use Supervisor, no automatic retry or overwrite.
Reporting merges this queue with the existing cohorts, preserving the Hammer
retry replacement rule. Archive the new cohort additionally to central
`runs/adroit-online-20261112/temp001/shard{0,1}` with checkpoint hashes.

## Hourly completion monitoring and stop-only shutdown — 2026-11-13

The user requested hourly monitoring, then explicitly changed the final action
to STOP ONLY: "다 되면 일단 반납하지말고 중지만 시켜놔". The earlier authorization
to destroy/return is revoked. Scope: Vast60326900 (175.155.64.205:19159, four5090s)
and60327673 (179.255.127.236:31020, four5090s). Each host has32 logical runs after
merging the baseline/migration, T0.01 and Hammer recovery queues. The preserved
failed Hammer attempt is history, not an additional logical completion target.
Both instances must finish all64 logical runs, including pending SAC and iBOLT,
before stopping. The16 completed dilg runs remain on central storage; no other
instances or dilg workloads are shutdown targets.

The read-only monitor runs every3,600 seconds in the active task, records both
servers' observations, and exits only when completion gates pass. No recurring
Codex automation is created. Unreachable servers, failed jobs, changed scope or
unrelated GPU processes block stopping. Verify32 completed3,000,064-step runs per
host, expected update counts,1,000-episode final evaluations and final model/
optimizer/replay files. Preserve all active learners and their original commits.

Before stopping, finish a stable central archive and compare SHA256 manifests
for all scientific outputs, checkpoints, failed attempts, source snapshots and
logs. Keep credentials/venvs out of the archive; preserve runtime version metadata.
The older uncompressed backup relay may be replaced with the committed compressed
relay, preserving its inventory and all source/destination files. Defer files
modified in the last120 seconds to avoid live logs invalidating large transfers.
This changes archival operations only, not any running numerical source.

Final STOP is a separate explicit operation after verified backup and a fresh
identity/completion check. Preserve both instances and their disks; never issue
DELETE/destroy/recycle. Reuse each instance's existing per-instance API
authentication without printing credentials. Record provider responses and verify
stopped status; a lost SSH connection alone is not proof of stopping. References:
https://vast.ai/developers/api and the Vast Docker environment documentation at
https://github.com/vast-ai/docs/blob/main/guides/instances/docker-environment.mdx.
