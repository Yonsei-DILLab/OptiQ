# MEOW action-saturation diagnosis on wall-free 4-Way (2026-09-26)

The native CleanRL MEOW baseline (alpha 0.2) is preserved. Its completed
seed-0/100k policy reaches a goal in 97/100 sampled rollouts, but nearly every
first horizontal action is saturated at +1 and the trajectories take a long
detour. The pinned flow's `get_qv` multiplies its implicit Q and V by alpha.
Because the 4-Way reward includes `-30*||action||^2` and negative squared
distance each step, alpha 0.2 may be poorly scaled for this task. This is a
diagnostic hypothesis, not a proven cause.

Run three independent MEOW-only alpha settings: 2, 10, 30. Change nothing else:
the exact wall-free symmetric 4-goal environment, native flow architecture,
sigma bounds [-5,-0.3], Adam Q learning rate 1e-3, target tau 0.005, gradient
clip 30, seed 0, 16 vector environments, batch 256, 16 learner updates per
16 transitions, 1024 transition warmup, 100k transitions total. The native
alpha 0.2 control remains the original frozen run. Evaluation samples the
policy directly from the same fixed center start; it does not use a selected
best-of-K action or alter the reward. Run an actual 1040-transition/16-update
preflight before each fresh 100k run. Save raw rollouts and Q/action probes
every 20k, final replay/checkpoint, source SHA, and final 100-episode goal
counts, success, mean path length and first-action saturation.

Choose only among completed runs. Favor success at least 95/100, then lower
first-action saturation and shorter paths. If no candidate improves the
trajectory without losing success, report that result rather than replacing
the native baseline. These tuned MEOW runs are separate ablations and must
never be silently substituted for the native MEOW results in 4-Way or N-Way
method-comparison tables.

Three guarded jobs wait for GPU locks and zero foreign compute PIDs on
vast-heechan-199 GPUs 0, 1, 2. They must not interrupt the long PointMaze
training or the already-queued N-Way baseline shard on GPU 3. Each job is
independent, with no automatic restart, retry or hyperparameter extension.
The exact frozen learning source and scheduling commit are recorded in the
campaign manifest before any preflight or main learning begins.
