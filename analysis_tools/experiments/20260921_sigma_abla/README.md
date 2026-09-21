# Initial sigma ablation, 2026-09-21

Base: direct-gmm-trg, fc6067b. Use its dedicated
analysis_tools/experiments/20260920_truncated_mll implementation unchanged.

Initial Gaussian scale sigma: 0.1, 0.2, 0.01, 0.05 (not log sigma).
Set initial_log_std=ln(sigma); zero-initialized scale-head kernel preserves the
exact requested starting scale. The truncated distribution's realized standard
deviation can differ near action bounds. Sigma remains learned after initialization.

Ant-v4, Walker2d-v4, Humanoid-v4, Hopper-v4, HalfCheetah-v4; seed 0 each,
20 runs, 10,000 environment steps. Existing 5,000-step warmup retained, so only
5,000 training updates. No early stopping or automatic restart.

Preserve branch training defaults: 64 mixture components and 64 candidates,
continuous normal latent, temperature 0.25, batch 256, UTD 1, policy delay 1,
Adam 3e-4 actor/critic, 256x2 networks, no clipping, bounded tanh centers,
truncated Gaussian sampling and likelihood, log_std range [-5,-1], no extra
behavior noise/uniform replacement. Only initialization varies across conditions.

Evaluation frequency 1,000 steps (plus initial step 1), 10 episodes each for
zero_z and stochastic_z with paired episode seeds and isolated evaluation RNG.
Report separate curves and final 10k returns; compare post-warmup reward AUC
from 5k to 10k. One seed and 5k updates cannot establish a general best sigma.
W&B OptiQ/abla, groups ENV_trg_sigma_10k_s0_20260921.

Commit this protocol and wrappers before training. Use immutable clean clones
and record source SHA in queue state and W&B provenance. On each of hosts 180
and 199, GPUs 0/1/2/3 map to sigma .1/.2/.01/.05 respectively. Host 180 queues
Humanoid, Ant, HalfCheetah; host 199 queues Walker2d, Hopper.
Use existing run-gpu.sh and user Supervisor; old campaigns stay stopped.
