# User-requested DACER priority, 2026-09-21

Stop the previous Ant/Humanoid controllers and cancel unfinished temperature
runs. Preserve all old sources, checkpoints, logs and job records. Select T
using completed common seed IDs across each environment's two candidates,
ranking stochastic_z reward over 900000 < steps <= 1000000 (20 x 10 episodes).
This explicitly supersedes the previous five-seed completion gate.

Run DACER enabled, temperature .25, beta 1, seeds 0..4 in both environments
first in assignment order. Immediately backfill free GPUs; once all five DACER
seeds are assigned, beta jobs may overlap the remaining DACER runs. Next run
beta .5/.9 seeds 0..4 at the selected T with DACER disabled. Reuse completed
beta results; restart interrupted beta attempts fresh for 1M steps, retaining
the partial attempts. Checkpoints do not contain complete replay/RNG state.

Unchanged defaults: actor/critic 256x2, random normal latent, N=M=64,
log sigma [-5,-1], initial log sigma -1, batch 256, UTD 1, LR 3e-4,
warmup 5000, 1M total steps, eval every 5000 with 10 episodes per mode.
DACER defaults: target -.9 * action_dim, initial alpha .27, alpha LR .03,
update every 10000 learner updates (also first update), GMM K=3 with 200
actions for each replay state. Behavior noise scale .15 Humanoid/.1 Ant.
Teacher, TD target, and mu-only zero_z/stochastic_z evaluations are unchanged.
W&B: OptiQ/gmm-trg. No other environments or ablations.

Commit code/protocol before launch. DACER uses the new frozen snapshot; beta
uses original snapshot 84f1e0a884349d6c4b0dae521839a8d4e5f46437.
Use supervisor and one GPU lock per worker. No automatic restart of failed or
interrupted jobs. Isolate pinned sklearn 1.5.2/joblib 1.4.2/threadpoolctl 3.5.0
under campaign deps. Do not mutate the shared runtime environment.

Completion requires final actor/critic checkpoints, config identity and the
complete finite 20x10 final evaluation window in both modes. A known W&B
post-training artifact upload timeout can be imported with an explicit warning
when these checks pass; retain the original failure record. Report sample SD
across seeds and seed counts. Compare beta=1 on common completed seeds and
also show all five seeds for new beta/DACER runs.
