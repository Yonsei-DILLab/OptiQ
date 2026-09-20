# Initial sigma sweep with log scale ceiling 0

User stopped the earlier [-5,-1] sweep and requested the same experiment with
log_sigma in [-5,0], checking whether scales stick to the new upper bound.
All new runs start from scratch. Old checkpoints, evaluation data, and logs stay intact.

Initial sigma .1, .2, .01, .05; five tasks (Ant, Walker2d, Humanoid, Hopper,
HalfCheetah), seed 0; 10k environment steps with the default 5k warmup and 5k
learning updates. W&B OptiQ/abla; groups ENV_trg_sigma_max0_10k_s0_20260921.

Preserve branch training settings: N64 x M64, normal random latent, T=.25,
batch256, UTD1, policy delay1, Adam3e-4, 256x2 GELU actor/critic, no clipping,
box-truncated Gaussian with bounded tanh centers, direct marginal NLL.
Only initial scale and the user-requested upper bound change.

Evaluation every1k, 10 episodes each for zero_z and stochastic_z; paired seeds
and isolated evaluation RNG. Diagnostics every1k: log_sigma min/mean/max,
sigma min/mean/max, and fraction of sampled outputs at the configured cap.
The fraction counts B x N x action-dimension outputs within 1e-6 log units of
the cap; a single observed maximum of 0 is not evidence all outputs are stuck.
The policy scale parameter is before truncation, not the realized action std.

Fix existing boundary diagnostics to read actor model log_std_min/max instead
of hardcoded -5/-1. This alters only auxiliary metrics, not loss or gradients.
Preflight verifies all20 configs, actual initial scales, and the real JIT
metric at raw log sigma -.5 (0% at cap0) and +.5 (100% at cap0).

Commit exact source and launch protocol before training; freeze clean clones.
Use the same two-host/eight-GPU queue and existing Supervisor. GPUs0..3 use
initialsigma .1/.2/.01/.05. Host180: Humanoid,Ant,HalfCheetah;
host199: Walker2d,Hopper. No automatic reruns. Final checkpoints and both
evaluation modes must reach10k before marking a run scientifically complete.
