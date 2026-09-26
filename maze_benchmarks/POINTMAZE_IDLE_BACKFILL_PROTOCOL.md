# Fill verified idle PointMaze GPU slots (2026-09-26)

The user requested existing pending jobs in empty GPU slots. This plan transfers
six already-authorized baseline jobs from46/199 to vast-heechan-6. Preserve all
running corrected DIPO and baseline workers. MFPO/MEOW multiseed remains held.
Learning source bfb06944705685a7dea84558aba40500c22349b9 remains unchanged:
256env,batch4096,16updates/256,8192warmup,1,000,192steps,62,000updates. The
DIPO32/2048 ratio does not apply to these other baselines.

IMPORTANT correction: preliminary commit098e22a planned host180 slots and
prematurely documented a passed compute check. Actual PyTorch CUDA initialization
failed on all three apparently idle GPUs0/2/3 (GPU1 already reports Unknown
Error). None of those jobs was transferred or launched. This replacement plan
excludes host180 and withdraws the associated guard/controller changes. No
driver or running process was changed. Do not launch the superseded plan.

Under each original46/199 queue.lock, assert the three chosen jobs are still
pending, record original configuration and mark them transferred before the
new destination workers claim them. If a job started, stop and replan rather
than duplicate or interrupt it. Save transfer sidecars with source/plan commits.

Destination root basename: pointmaze-baselines-idle-backfill-20260926 on host6.
GPU1/2 are idle; start two jobs immediately after independent preflights. Guards
on0/3 wait for corrected Simple/4-Way DIPO to finish before backfilling. Keep
existing locks, compute-PID checks and20-second idle checks. No learning change.
Use a committed controller snapshot and frozen learner bfb069, validate each
8448step/16update preflight, retain200k/200episode and final500episode evaluations,
raw trajectories/checkpoints/figures. Failure pauses destination pending work
and preserves live jobs; no automatic retries.

Collect source/config/proofs/evaluations with SHA. Track six transferred and18
remaining/completed jobs once each, total24 baseline multiseed additions.
Original queue transferred entries are provenance, not additional experiments.
