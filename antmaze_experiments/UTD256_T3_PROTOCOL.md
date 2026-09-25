# OptiQ AntMaze T=3 UTD=1 v3/v4 test (2026-09-25)

The `basic_euclidean` T=1 seed-0 control on v1-v4 was stopped as a set at the
user's request after v3 had zero successes in 40 fixed-start sampled-policy
episodes at 100,096 transitions and v4 had zero in 40 at 300,032 transitions.
Preserve its source commit `df70ccf94d268478314a5dc23a2fc5d724549ce0`,
partial evaluations, checkpoints, and logs. Stopping at these points is an
early screen, not a completed-budget performance comparison. Other reward-grid
campaigns are independent and remain untouched.

Start two *fresh*, independent OptiQ runs: v3 seed 0 on vast-heechan-180 and
v4 seed 0 on vast-heechan-199. Relative to the corresponding T=1 control job,
the only learning configuration change is fixed teacher temperature **1 → 3**.
The new job identifier and source commit differ solely for provenance.

Keep reward `100*(nearest-goal Euclidean d_current-d_next)` with no success
bonus or step cost. Keep original fixed full-state training/evaluation starts
for v3/v4, original success/termination, 256 parallel environments, batch
4096, 256 learner updates per 256 new transitions, replay capacity 1M, 8,192
warmup, gamma .99, tau .005, 256×2 actor/twin critics, actor/critic Adam
3e-4, mean-head scale 1e-4, DACER and NovelD off, random latent, N=M=64,
beta 1, log sigma [-5,-1] initialized at -1, and unchanged teacher proposal.
Use the upstream post-warmup budgets (v3 4M, v4 5M). Evaluate every 50k total
transitions with 40 fixed-start episodes each for the random-z mu-only policy
and the sigma-included directly sampled policy. Save intermediate policy-only
states; at the final budget save 100 episodes per mode and full state. W&B is
`OptiQ/antmaze`.

Commit and push this protocol and registration script before launch. Freeze
the full commit on both servers. The per-job preflight must verify actual
batch-4096 updates, 256:256 collection ratio, reward and fixed reset before
main training. Report observed success and route selection at the first 50k
evaluation. Do not infer a T=3 improvement from zero-success early screens.
Do not use vast1 or relaunch the stopped T=1 campaign.
