# Wall-free N-Way mode-scaling queue (2026-09-26)

The user requested 8/12/16/32-Way experiments after the 4-Way baseline
comparison. These are **new symmetric wall-free point tasks** using the 4-Way
movement and reward equations. They are not the walled ICML Four-Way task.

For each N in {8,12,16,32}, N goals are equally spaced counter-clockwise on a
radius-6 circle, starting at east. Every task starts at the origin, observes
`(x,y)`, accepts displacement actions clipped to `[-1,1]^2`, clips positions to
`[-8,8]^2`, and truncates after 24 actions. At the new position,

`r = -30 ||a||² - min_i ||s' - g_i||² + 10 · success`.

Success terminates within radius 0.5 of the nearest goal. The smallest adjacent
goal separation is 1.176 at N=32, so success discs do not overlap. This shared
radius-6 / success-0.5 / horizon-24 family isolates goal count across the four
new tasks. Its geometry differs from the historical radius-5, success-1,
horizon-20 4-Way task; absolute returns are not directly comparable.

The first registered four jobs are OptiQ Direct GMM/iBOLT, one seed 0 per task.
Other algorithms are added only if the user selects all baselines for this
N-Way expansion. OptiQ uses T=1, N=M=64, random Gaussian latent, 256×2 actor
and twin critics, direct GMM NLL, mean-head init scale 1e-4, log sigma [-5,-1]
with initial -1, density correction on, DACER off. Each job has 16 vector
environments, replay 1M, batch 256, 16 learner updates per 16 collected
transitions (UTD=1), actor/critic Adam 3e-4, gamma .99, tau .005. Warmup 1024
is included in the 1,000,000 environment transitions; exactly 998,976 learner
updates follow it. These match the 4-Way study profile apart from the explicit
N-Way geometry and 1M budget.

Each job has a real 1040-step / 16-update preflight before fresh main learning.
At every 100k transitions save the sampled-policy and, for OptiQ, mu-only raw
trajectories; 256 evaluation episodes per mode; an action/Q grid; a policy/critic
checkpoint; and a four-panel PNG with nearest-goal distance contours, sampled policy arrows,
rollouts, learned policy-averaged Q and exact per-goal counts. At 1M, evaluate
1024 episodes per mode and save the final replay too. Evaluations are processed
in batches of at most 128 episodes to bound GPU memory. The policy/sigma-included
rollout is the main mode-diversity measure; mu-only is separate. Goal contours
are the position-dependent reward term, **not** full reward or learned policy
density. The absolute critic values
of different algorithms need not be comparable.

One frozen source commit, manifest SHA256 checks, GPU flock locks and supervisor
queues govern launch. An occupied GPU leaves its assigned job waiting; jobs on
other slots are independent. A failure stops that shard's next task without
deleting partial results or automatically restarting. The 1M checkpoint is a
policy/critic/replay artifact, not a guaranteed exact optimizer/RNG resume;
extension beyond 1M needs its own documented plan after reviewing 1M coverage,
success and Q quality. Do not treat an independent longer run as continuation.
