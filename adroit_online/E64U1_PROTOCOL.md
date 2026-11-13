# iBOLT env64/UTD1 Door and Hammer — 2026-11-13

The user explicitly requests Door/Hammer x actor temperature0.05/0.10/0.25 x
seeds4/5, each3M, on heejoonorm@175.155.64.205:19159 (Vast60326900).
These12 runs start from fresh parameters and empty online replay. This authorizes
only this new cohort following the old campaign's stop. Never restart old queues.
Use branch heejoon-rebuttal and commit before any GPU smoke or full learning.

## Scientific configuration

Reuse the numerical learner, native AdroitHand{Door,Hammer}Sparse-v2, DACER and
all evaluation conventions in PROTOCOL.md. Relative to the same-temperature
env256/UTD0.25 iBOLT configuration, change only num_envs256->64, UTD0.25->1 and
derived identifiers/counts. Each vector collection still precedes64 updates.
Keep four collection workers, replay1M, batch256, warmup10,240, gamma0.99,
tau0.005, lr3e-4, GELU256x2, N=M=64, logstd[-5,-1], density correction beta1,
mean proposal twin-Q, actor update every critic update, and unchanged DACER.
DACER's10,000-update interval stays in learner-update units, so its cadence per
environment transition changes with UTD. No demonstration, BC, HER or shaping.

Retain3,000,064 training transitions, matching completed baselines, with
2,989,824 critic and actor updates. Requested budget is3M; the64-transition
rounding is shared by all methods. Initial/random evaluations use100 episodes,
regular evaluations every50,176 use100, final evaluation uses1000 episodes.
Checkpoints every250,880 and at final; preserve final replay and auxiliary state.
Use identical seed-paired evaluation resets. Primary iBOLT uses stochastic_z;
SAC/TD3 primary is deterministic. Keep zero_z and full mixture as secondary modes.
Report any-success and cumulative>=25 successful steps separately; the latter
does not require consecutive successes. Evaluation interactions are not training.

## Execution and validation

New root: /workspace/heejoon-adroit-ibolt-e64u1-20261113.
Reuse /workspace/heejoon-adroit-online-20261112/venv unchanged; record installed
package versions and native robotics VCS identity. Deploy a git-archive source
with full commit SHA and file hashes. Controller plan, each smoke/full PID, GPU
UUID, CPU affinity, resolved config and result paths carry provenance.

Use GPUs0/2/3, at most2 jobs each, eight disjoint logical CPUs/job. The user also
explicitly asks to recheck GPU1. Run identical fresh integer arithmetic checks
(JAX and Torch versus NumPy) and native Hammer T0.10 env64/UTD1 smoke learning on
GPUs0/1. Add GPU1 only if BOTH device checks pass; exclude any failing device.
Record GPU1's prior unresolved diagnostic failures and the limits of a short
validation; this does not prove a hardware defect or complete resolution.
Require at least12GiB free before
admission and reject unrecognized compute PIDs. Make all tasks independently
eligible; seed/temperature ordering is scheduling only. Stagger process starts.
Use original XLA execution flags, no abandoned synchronization/compiler fixes.
No package/driver changes. Preserve other processes, old data and old services.

Before every full run, perform the native task/temperature preflight on its
assigned GPU with the new profile:53,248 transitions,51,200 warmup,2,048 updates.
The diagnostic seed is4, separate from the full run's assigned seed4/5. Require
JAX GPU arithmetic, native seeded physics/reward/timeouts, empty replay,
evaluation RNG preservation, exact counts, finite checkpoint states and exact
actor/critic optimizer restore. The smoke does not transfer data/weights to the
full run. Check the first full DACER update boundary before reporting healthy
learning. No automatic retries or overwrites; failed evidence remains. A failed
process quarantines that GPU from subsequent starts pending review, without
killing peers. Supervisor autostart/autorestart are false; SIGTERM drains active
children rather than killing learner groups. This queue never stops the instance.

## Matched comparison and storage

Reuse exactly8 completed baseline runs from source
e1a244a8216d3286fee1dbcc84fd3adfb748dabb:
{door,hammer}-{sac,td3}-s{4,5}-e64-u1.
Their central roots are
/data1/heejoonorm/OptiQ/runs/adroit-online-20261112/shard{0,1}/runs.
Shard0 has SAC seed4 and TD3 seed5; shard1 has SAC seed5 and TD3 seed4.
Verified raw evaluation counts and seed schedules are indexed by the previous
final-results/verified-results.json. Do not substitute env256 curves or best
checkpoints. Plot each task/seed at common environment steps, separate any-success
and>=25 metrics, and report final3M counts/1000. Partial curves end at their actual
latest step; no extrapolation, smoothing or result-based run selection.

Archive source, protocol, runtime, plan/PIDs, logs/evaluations and checkpoints to
dildata:/data1/heejoonorm/OptiQ/runs/adroit-ibolt-e64u1-20261113/shard0.
The local e64u1_archive module reuses the existing compressed SSH relay every
five minutes, defers recently modified files and verifies checkpoint hashes.
It uses existing local SSH authentication; no key copying or access-control edits.
The read-only collect_curves --profile e64-u1 and compare_e64u1 modules produce
seed-specific plots and an audit against the verified old baseline index.
Keep data/credentials/venvs out of Git. Preserve original server files; do not
delete or claim complete backup before hashes verify. No new SAC/TD3, T0.01,
Pen/Relocate, confirmation seeds0-3, temperature selection or other hosts.
