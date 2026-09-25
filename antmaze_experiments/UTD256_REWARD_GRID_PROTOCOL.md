# OptiQ AntMaze UTD=1 reward-only queue (2026-09-25)

Continue the running `basic_euclidean` four-task control unchanged. Queue four
fresh reward variants for each of v1–v4, one training seed (0) per task and
variant. `d` is nearest-goal Euclidean distance measured at the true current
and next simulator XY. There is no discount inside any reward formula:

| Variant | Reward at each transition |
|---|---|
| `progress10` | `10 * (d(current) - d(next))` |
| `progress100_cost01` | `100 * (d(current) - d(next)) - 0.1` |
| `progress10_cost01` | `10 * (d(current) - d(next)) - 0.1` |
| `negative_distance` | `-d(next)` (existing `dense` profile) |

No variant adds the upstream sparse success bonus. All preserve its physical
success condition and termination. v1 keeps upstream random starts; v2–v4 keep
their original fixed full-state start in both training and primary evaluation.
The existing control is `100 * (d(current)-d(next))`, with no step cost.

Each queued job inherits the live `basic_euclidean` profile exactly except its
reward: OptiQ 256×2 actor/twin critics, mean-head scale 1e-4, actor/critic
LR 3e-4, T=1, DACER off, NovelD off, random latent, N=M=64, beta=1,
log sigma [-5,-1] initialized at -1, gamma .99, tau .005, replay 1M;
256 parallel environments, batch 4096, 256 optimizer updates per 256 new
transitions (UTD=1), warmup 8192. Native post-warmup environment budgets are
v1/v2 3M, v3 4M, v4 5M. The 8,192 warmup transitions are also counted in
total steps. One GPU runs one job at a time.

Evaluate every 50k total transitions (rounded to 256-transition boundaries):
40 episodes each for random-z mu-only and direct policy with conditional sigma.
Save evaluation-only intermediate policies. At completion, evaluate 100 episodes
per mode and save full replay/optimizer/RNG/simulator state. W&B is
`OptiQ/antmaze`. Each job first passes the real 256-env, batch-4096,
256-update preflight. The controller checks reward formulas and replay,
architecture, LRs, counters and saved state before accepting it.

The two RTX5090 hosts independently queue eight jobs each. The existing
control is the priority predecessor: its running GPUs are reserved while
the other eligible unlocked GPUs are filled immediately. On each completion,
backfill the next job within the controller's two-second cycle. There is no
all-maze or all-variant completion gate. A failure holds that host's pending
queue, preserves live workers and logs, and requires diagnosis before any
restart. The reserved RTX4090 `vast1` is never used for AntMaze. Commit, push,
share, and pin the exact frozen source before launching jobs.
