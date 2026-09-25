# Fixed-start AntMaze dense reward, gamma 0.9

This experiment replaces the two running gamma=0.99 jobs in
`antmaze-optiq-fixed-start-nearest-dense-v34-1m-s0-20260925-r2` at the user's
request. Preserve their partial results and do not resume them. The controls
used frozen source `c3a2668cc180d4713204181ec09b87aecbe7e52e`.

Run one fresh OptiQ seed0 policy each: v3 on `vast-heechan-180` GPU3 and v4 on
`vast-heechan-199` GPU3. The *only learning-setting change* relative to the
cancelled controls is the discount factor, gamma 0.99 to 0.9. The TD target
uses that discount; this changes the objective and its effective planning
horizon. It is not a resume or a reward rescale.

In particular, keep both maze goals fixed, original fixed full-state origin on
every training and evaluation reset, and reward
`r_t = -min_g ||p_{t+1} - g||_2` with no step penalty, success bonus, or NovelD.
Keep success termination, T=1, DACER off, random latent, N=M=64, log sigma
`[-5,-1]` initially `-1`, 256x2 GELU actor/twin critics, actor/critic Adam
3e-4, tau 0.005, replay capacity 1M, 256 vector environments, batch 4096,
and 256 learner updates per 256 collected transitions. Warmup 8,192 is included
in a 999,936-transition total budget per task.

At each 50k evaluation, save 40 episodes each of random-z mu-only and direct
random-z plus conditional-sigma policy from the same full start. Record route
counts/proportions, successes, returns, and raw XY trajectories. Save policy
snapshots at these evaluations, and at the final step save full state and 100
episodes per mode. Log to W&B `OptiQ/antmaze`. Do not include the cancelled
control's partial trajectories as completed gamma=0.99 results.

Use the immutable committed source as the service working directory. Each job
must pass the existing 8,448-transition/256-update preflight, including a
readback of the model gamma=0.9 and dense replay reward audit, before main
training. No other job or GPU slot is affected.
