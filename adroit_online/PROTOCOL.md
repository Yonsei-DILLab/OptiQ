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
