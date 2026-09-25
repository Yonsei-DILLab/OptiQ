# OptiQ/iBOLT AntMaze UTD=1 experiment (2026-09-25)

Run one fresh seed-0 policy on each of DDiffPG AntMaze v1-v4 to test whether
the 8-update collection schedule limits critic/actor adaptation. Keep prior
results and frozen sources intact. The selected `condition` is written into
each source manifest before launch; only one condition is registered.

## Single changed training setting

The explicit `env256-update256` collection profile gathers 256 transitions
from 256 parallel environments, then performs 256 learner updates with
batch 4096. This is **one optimizer update per new environment transition**.
The previous AntMaze controls used 8 updates per 256 transitions (UTD=1/32).
The new profile is opt-in; the default collection profile remains unchanged.
Replay sample draws per new transition rise from 128 to 4096. Replay capacity
remains 1,000,000 transitions, and warmup remains 8,192 transitions. Higher
UTD also increases actor changes during one environment episode, so route
retention is measured, not presumed.

The two selectable conditions are:

| Condition | Reward | OptiQ model and optimizer | Prior comparison |
|---|---|---|---|
| `control` | `100*(nearest-goal Euclidean distance decrease)`; zero step penalty and bonus | historical 256x3 actor/twin critics, mean-head scale 1, actor LR 3e-4, critic LR 5e-4 | existing v3/v4 route-collapse controls at UTD=1/32 |
| `basic` | original DDiffPG sparse goal reward; NovelD disabled | restored OptiQ 256x2 actor/twin critics, mean-head scale 1e-4, actor/critic LR 3e-4 | restored basic OptiQ profile; no matching completed 8-update run is claimed |
| `basic_euclidean` | `100*(nearest-goal Euclidean distance decrease)`; zero step penalty and bonus | restored OptiQ 256x2 actor/twin critics, mean-head scale 1e-4, actor/critic LR 3e-4 | user correction after the sparse run; prior Euclidean controls also differ in model/init/LR |

All conditions use T=1, DACER off, NovelD off, random normal latent at every action,
N=M=64, beta=1, Direct GMM marginal NLL, log sigma bounds [-5,-1]
with initial -1, gamma .99, tau .005 and replay 1M. Ant physics, 8D
actions in [-1,1], original success/termination and starting states remain
upstream. v1 training/evaluation starts randomly; v2-v4 start from the
original fixed full state. No `vast1` resources are used.

## Budgets, evaluation and provenance

The upstream post-warmup budgets are v1/v2 3M, v3 4M, v4 5M environment
transitions. The run counter also includes the 8,192-transition warmup and
stops after the native budget is exceeded on a 256-transition boundary.
Each task has one job on either vast-heechan-180 (v1,v3) or
vast-heechan-199 (v2,v4), with independent GPU-lock-aware backfill.
No cancelled campaign is resumed.

Every 50,000 *total* environment transitions (rounded up to a multiple of
256), evaluate 40 episodes per policy mode: direct stochastic policy
(fresh z plus conditional sigma) and random-z mu-only (`native`). Save the
evaluation-only actor/critic policy state at those checkpoints so route
loss can be inspected afterwards. Final evaluation uses 100 episodes per
mode plus zero-z, and saves the full replay/optimizer/RNG/simulator state.
Report each task's 50k boundary with actual global/total steps, learner
updates, success rate, goal/path split when available, and training progress.
Zero successes or one sampled path are reported as observations, not
proof of zero probability. W&B is `OptiQ/antmaze`.

Before main learning, each job runs the real 256-environment preflight:
8,192 warmup transitions followed by exactly 256 batch-4096 OptiQ updates.
The controller verifies actual counters, architecture and optimizer LRs,
reward replay, unchanged basic defaults, final checkpoint readback and
evaluation settings. A failure holds pending jobs and preserves the logs;
there is no automatic retry or parameter change. Source is committed,
pushed and copied to both immutable `sources/<full SHA>` directories before
registration. The manifest and every result carry the full learning SHA.

The `basic` sparse campaign was stopped by the user before its first 50k
evaluation. Its source, status, preflights, partial logs and W&B run IDs are
preserved. The selected replacement is `basic_euclidean`, started fresh from
seed 0; it changes only the reward from that first registration while keeping
the restored basic OptiQ model and UTD=1. The preexisting Euclidean route-collapse
controls used the distinct legacy architecture and learning rates, so they are
not exact single-variable UTD controls.
