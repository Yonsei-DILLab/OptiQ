# AntMaze OptiQ N=M128/256 launch, 2026-09-25

Four requested seed-0, T=1, `100*(d_current-d_next)` Euclidean runs were
registered. Total budget per run is 1,000,192 environment transitions
(1M rounded up to a 256-environment boundary), including 8,192 warmup.
Batch is 4,096; every 256 new transitions trigger 256 actor and critic
updates. Actor and twin critics are 256x2, DACER and NovelD off, random z.
Evaluation uses official fixed starts for v3/v4, 40 episodes per mode every
50k, with final 100 per mode.

| Task | N=M | Host / GPU | Source | W&B |
|---|---:|---|---|---|
| v3 | 128 | vast1 / 0 | `b0df33807a05debd3df47536c34058b0962d6c8f` | https://wandb.ai/OptiQ/antmaze/runs/a75g805j |
| v4 | 128 | vast1 / 1 | `b0df33807a05debd3df47536c34058b0962d6c8f` | https://wandb.ai/OptiQ/antmaze/runs/y9rm277r |
| v3 | 256 | vast-heechan-180 / 1 | `e845c7b7276b393d2924d90d85454f7dbeaf32cf` | https://wandb.ai/OptiQ/antmaze/runs/99myg9av |
| v4 | 256 | vast-heechan-199 / 1 | `e845c7b7276b393d2924d90d85454f7dbeaf32cf` | https://wandb.ai/OptiQ/antmaze/runs/sbn2ntz2 |

Each real preflight reached 8,448 total transitions and completed 256
batch-4,096 actor/critic updates. Its config and initial/final proofs verified
the requested N, one proposal per policy sample, M=N, T=1, and 256
environments. All four main runs started under supervised GPU locks.

The original N=M256 full-array preflights (`b0df338`) failed with XLA GPU
out-of-memory at their first update; neither began main training. Their
failure files and source were preserved. The `e845c7b` source computes each
N=M256 actor gradient in 16 chunks of 256 states, averages the full effective
batch-4,096 gradient, and applies Adam once. It preserves the objective and
update ratio, although per-chunk PRNG draws are not bitwise the same as the
full-array realization. Critic updates are unchanged.

Before registration, the four Hopper GMM-TRG and three NM GPU workers on
vast1 were stopped by explicit user instruction. No checkpoint, source,
result, log, or system service was deleted. The vast1 canonical Git branch,
which had local commits ahead of GitHub, was left untouched. Both ongoing
T=3 v3/v4 5090 learners remain running.

Source and protocols are committed and pushed on `direct-gmm-trg-antmaze`.
The three-host policy is updated in `antmaze_experiments/HOST_POLICY.md` and
the workspace and repository `AGENTS.md` files.

## User stop, 2026-09-25

The user directed: if N=256 lacks memory, do not run it. The original
full-array N=256 preflights had already OOMed. Although the 256-state
gradient-accumulation implementation passed both preflights, its two main
runs and W&B sync services were stopped in response. GPU1 on each 5090 was
freed; no N=256 learner remains. Preserve both sources and all partial data.
Do not retry or resume N=256. The N=128 v3/v4 jobs on vast1 and the preexisting
T=3 v3/v4 jobs on the two 5090 hosts continue.

The thread heartbeat `antmaze-optiq-n-m` follows only the two N=128 runs
and stops after their validated final results are reported.
