# Log-sigma cap 0 versus -2, Ant and Humanoid, 20k

User stopped the previous 20-run sigma sweep and requested Ant/Humanoid
comparison of caps 0 and -2 through20k steps, with explanation of trajectories.
Four fresh runs, seed0, one per GPU on host180. No checkpoint resume.

Use initial sigma=.1 in both treatments to isolate the cap effect; this is
inside both ranges and was explicitly communicated before launch. Initial .2
would exceed exp(-2), confounding initialization with clipping. Lower bound=-5.
Compare upper bound0 (sigma ceiling1) versus-2 (ceiling0.1353352832).

20,000 environment steps including5,000 default random-action warmup:
15,000 actor/critic updates. Same64x64, continuous random normal latent,
T=.25, batch256, UTD1, policy delay1, Adam3e-4, 256x2 GELU, no gradient clipping.
Unchanged bounded tanh centers, box-truncated sampling and marginal NLL.

W&B OptiQ/abla. Both zero_z and stochastic_z conditional-mean evaluations,
10 paired-reset episodes/mode at step1 and every1k; evaluation RNG isolated.
Record scale min/mean/max and fractions at actual configured bounds every1k.
Fractions aggregate minibatch states x sampled latents x action dimensions;
use tolerance1e-6 in log units. A single maximum at the cap does not establish
that all outputs are saturated. Show occupancy trajectories and final values.
Compare reward at10k/20k and post-warmup normalized trapezoidal reward AUC.
Use one-seed language, not claims of general superiority.

Source inherits configuration-aware saturation metrics from31e84e8.
Commit code/config/launch protocol before running; keep previous snapshots and
logs intact. No automatic restart, new sigma conditions, or extra training.
