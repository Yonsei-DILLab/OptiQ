# PointMaze 1M/2M/3M fresh comparison

This experiment restarts all 21 PointMaze policies from seed 0. No previous
checkpoint, replay, optimizer state, or partial trajectory is used for training.
The earlier 100k/200k/300k runs and their frozen source remain intact.

Simple/Medium/Hard use the unchanged official DrAC maps, fixed center reset,
sparse +100 goal reward, and 150/300/600-step horizons. The training budgets are
respectively 1,000,000/2,000,000/3,000,000 environment transitions, rounded up
to a complete vector collection. Every maze trains OptiQ, SAC, SQL, MEOW, MFPO,
DIPO, and TD3 once at seed 0. OptiQ has T=3, N=M=64, random Gaussian latent,
256x2 actor/critics, log sigma [-5,-1], mean-head init 1e-4, and **DACER off**.
The other methods retain their previous native policy and optimizer settings.
All methods retain 256 parallel environments, batch 4096, 16 learner updates
per 256 transitions, replay capacity 1M, warmup 8192, discount .99, and the
same method-specific learning rates. No entropy, reward, or architecture
ablation is mixed into this budget comparison.

Each job passes its own real-update preflight using the previously validated
candidate profile before fresh training. Queue workers claim independent jobs
as each GPU becomes available. The shared GPU flock and hardware-idle guard
protect unrelated workers; no existing work is stopped. A failed job pauses
further claims on that host and is never silently restarted.

Directly sampled policy evaluation is run every 20% of each maze's budget with
200 episodes per ordinary and obstacle map. The final checkpoint uses 2,000
episodes per map, and OptiQ records 2,000 additional random-z mu-only episodes.
Evaluation does not add DACER noise. Final evaluation-only policy, full replay,
raw paths, goal IDs, step/update counts and config are retained. Goal-specific
sampled path panels and 2,000-episode visitation maps are post-hoc figures;
they distinguish equal-number illustrative paths from empirical goal frequency.

W&B is unchanged from the existing PointMaze pipeline (local verified artifacts
are authoritative). The frozen source SHA is recorded in queue/job/config and
must be committed and shared before any preflight. The published older paper
budgets were shorter; these runs are a new longer-budget experiment, not an
exact paper reproduction or a multi-seed result.
