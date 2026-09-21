# September19 resume after infrastructure failures

Resume the unchanged numerical study cc11f537af330e23e1cc77cb94a9426660b55ebb
at its original immutable root. Completed trials and checkpoints retain that
original commit and source ID. The operational worker has a separate commit.
Do not relabel old checkpoints with the operational code revision.

Exclude previously unsuitable nodes and additionally node05. Allocate one GPU,
two CPUs and32GB per worker; up to16 workers for unchanged numerical runs.
JAX GPU initialization and a synchronized matrix product must pass before any
trial is claimed. Stop a worker after any CUDA/backend failure instead of
claiming more trials. Preserve old failure markers under per-trial attempts;
only identified GPU failures or interrupted launches are re-enabled. Numerical
NaNs remain excluded. Never overwrite completed trials or remove checkpoints.

All legacy row-argmax closed-loop trials are excluded from the old revision,
including not-yet-started large sizes. A separate corrected immutable revision
handles these64 trials. Its N16/M64 subset replaces16 invalid first-step runs;
the48 larger conditions retain their original registered settings. This is a
zero-noise TD adapter fix, not a change of actor, proposal, loss or temperature.
Require independent TD target/loss/gradient checks and finite rollout plus
checkpoint-resume agreement in all four dimensions before production eligibility.
Up to4 additional workers service this correction after validation passes.

Use dynamic per-trial dependencies only. Existing explicitly held studies and
MuJoCo are outside this resume. Before ending the task, verify actual GPU work,
checkpoint advancement, and dildata backup. Do not wait for campaign completion.

## Scheduler-query resilience

A transient squeue error terminated five idle workers after completed trials.
Retry failed/timed-out/partial scheduler queries with10..60second backoff,
outside the claim lock. Never interpret a failed query as zero live workers;
never reclaim leases until a valid response includes the currently allocated
job. GPU preflight and numerical code remain unchanged.

Replace each original worker once. Already terminated workers are eligible
immediately; healthy workers keep training. Their replacements use afterany
on that concrete worker ID, so they activate only when the original terminates.
These are operational handoffs, not completion dependencies between environments
or algorithms. There are still at most16 main plus4 legacy concurrent workers.
An already submitted replacement is never duplicated on a retry of the launcher.
