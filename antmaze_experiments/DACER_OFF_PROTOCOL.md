# AntMaze OptiQ DACER OFF, T=1

The user requested on 2026-09-24 that the current target-entropy sweep stop and
four fresh OptiQ v1-v4 seed0 experiments run with DACER=false. The previous
two completed policies and all interrupted logs/checkpoints remain preserved;
the remaining target sweep is cancelled, including pending entries.

The only learning change relative to the fixed-T1 control
`7af193833466f5bc41853948d6f417fc6aaa6803` is `dacer.enabled=false`.
Temperature remains1, beta1, dense negative nearest-goal distance, NovelD OFF.
Disabling DACER removes the external behavior Gaussian noise and its entropy
estimator/alpha updates. Fresh latent and conditional Gaussian action sampling
remain active. It does not disable policy stochasticity or change sigma bounds.

Preserve actor/twin-critic256x3 GELU, actorLR3e-4/criticLR5e-4, Adam,tau.005,
mean-head variance scale1,randomz,N=M64,log sigma[-5,-1]/initial-1,plainTD.
Keep256CPU vector environments,batch4096,8updates per256transitions,replay1M,
8192warmup. Native budgets remain3M/3M/4M/5M excluding warmup, with the original
strict-greater/vector-rounding accounting. No additional seeds or algorithms.

Two independent jobs per server:180 v1/v3,199 v2/v4; GPU locks and2second
backfill are retained. No maze completion barrier. Each job first passes the
actual8448transition preflight with8updates and checkpoint/replay readback.
OFF verification checks the live model flag, absent regulator state, bit-exact
train-versus-direct-policy actions at the same restored RNG, and zero external
noise/regulator updates. Main runs start fresh after their own preflight.
Failures hold pending jobs; no automatic restart. Commit/push/share frozen
source before launch. Campaign `antmaze-optiq-dense-dacer-off-T1-s0-20260924`,
W&B `OptiQ/antmaze`.

Evaluate every250k with40episodes per native/direct mode,random xy[-2,2]
starts,original pose/velocity. Save OptiQ evaluation-only policy checkpoints
at evaluations. Final100episodes per native/policy/zero_z mode, each with
random and fixed-full-state starts, and a verified full replay/model/RNG state.
No external behavior noise or intrinsic reward in evaluation. Preserve raw
trajectories, failures and initial states, and do not pool policies as diversity.

Entropy-target calibration is a separate read-only analysis of saved models
and logs. No positive entropy target, alpha learning-rate/interval, actor sigma
bound, teacher temperature or baseline-training changes are authorized here.
