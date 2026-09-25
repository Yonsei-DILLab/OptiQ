# Fixed-start, fixed-goal-set AntMaze dense-reward runs

This small OptiQ-only comparison uses v3 on `vast-heechan-180` and v4 on
`vast-heechan-199`, using GPU 3 on each RTX 5090. Other jobs and their GPU locks
remain untouched; `vast1` is excluded.

Each task resets to the original full state at the maze origin. Both goal
coordinates supplied by the maze remain fixed for the entire run and every
episode; there is no random goal selection and neither goal is removed:

| Maze | Fixed goal coordinates | Route categories |
| --- | --- | --- |
| v3 | `(-12, 12)`, `(12, -12)` | left, right, both, uncommitted |
| v4 | `(-16, 4)`, `(-16, -4)` | upper, lower, uncommitted |

The dense reward is the existing nearest-goal Euclidean distance penalty,
`r_t = -min_g ||p_{t+1} - g||_2`, using post-transition XY and the two fixed
goals. It has no per-step cost, success bonus, or NovelD. The maze's original
success termination behavior is retained.

Both jobs use OptiQ's basic 256x2 actor/twin-critic configuration, `N=M=64`,
temperature 1, DACER off, fresh random Gaussian latent, actor log-std bounds
`[-5, -1]` with initial `-1`, 256 parallel environments, batch 4096, and 256
learner updates per 256 collected transitions (one update per transition).
Replay capacity is 1M; the 8,192-transition warmup is included in a total budget
of 999,936 transitions, the largest vector-aligned total not exceeding 1M.

Every 50,000 transitions (rounded up to a 256-transition collection boundary),
evaluate 40 episodes from the same fixed full state in each of two modes:
OptiQ's native random-z mu-only output and direct policy actions with conditional
sigma. Save route counts, proportions, success counts, returns, and raw XY paths
per mode. Log route counts and proportions to W&B. Save policy-only snapshots
at each interim evaluation and one full training checkpoint at the final step.
The final evaluation uses 100 episodes per mode. Route proportions summarize
observed path categories from one seed; they are not a claim of causal latent
mode assignments.
