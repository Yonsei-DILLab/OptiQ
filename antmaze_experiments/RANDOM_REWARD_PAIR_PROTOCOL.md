# Random-start OptiQ reward pair, 2026-09-25

Stop and preserve the previous T=3 v3/v4 work, the random-start N=M64
`100Δd` v3/v4 work, and the fixed-start N=M128 v3/v4 work. Do not resume any
of them. Train fresh seed-0 OptiQ policies for v3 and v4 under two rewards:

| Label | Environment reward |
| --- | --- |
| `progress100_cost01` | `100*(d(current)-d(next)) - 0.1` |
| `negative_distance` | `-d(next)` |

`d` is nearest-goal Euclidean XY distance; both rewards omit success bonus and
NovelD. Keep upstream success termination and time limit. The -0.1 cost matches
the earlier approved `progress100_cost01` reward-grid definition. It is applied
per simulator step, not per 256-environment vector step.

Both mazes use upstream `random_init=True` at every training and primary
evaluation reset, drawing XY in [-2,2]. This changes their native fixed resets.
A fixed-full-state evaluation is supplementary; compare same-policy
route choices from that evaluation when assessing multimodality. Do not pool
trajectories from different learned policies or attribute start-state diversity
to action multimodality.

Hold the basic OptiQ profile: T=1, N=M=64, fresh Gaussian latent, beta=1,
DACER/NovelD off, actor/twin critic 256×2 GELU, mean-head scale 1e-4,
log sigma [-5,-1] initially -1, Adam actor/critic 3e-4, gamma .99, tau .005,
256 parallel environments, batch 4096, 256 learner updates per 256 collected
transitions, replay 1M, warmup 8192. Native post-warmup budgets are v3 4M and
v4 5M. Evaluate every 50k total transitions with 40 episodes/mode for
random-z mu-only and direct policy including conditional sigma; save evaluation
policy checkpoints. Final evaluation has 100 episodes/mode/reset and a full
checkpoint. W&B project is OptiQ/antmaze.

Run v3 on vast-heechan-180 and v4 on vast-heechan-199. Each server gets
two independent jobs and uses existing GPU locks with two-second backfill.
`vast1` receives no new AntMaze work. Complete a real 256-env/batch-4096,
256-update preflight for each job before main learning. Verify reward formula,
random reset spread, N=M64, saved state, and exact configuration. Commit/push
and freeze the source on both servers before launching. Preserve logs and
partial data from cancelled campaigns. Failed jobs hold pending entries, and
must not be restarted automatically.
