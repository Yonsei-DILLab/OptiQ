# Baseline configuration audit before the longer PointMaze run

The previous seed-0 Simple/Medium/Hard runs all completed with 256 environments,
batch 4096, and 16 updates per vector collection. The new campaign keeps those
settings and changes only the total environment-transition budget and final
evaluation count. The actual previous `config.json` files and the adapter source
were checked, not just the queue plan.

| Method | Actual previous policy/learning profile | Task compatibility |
| --- | --- | --- |
| OptiQ | Direct GMM/TRG, 256x2, N=M64, T3, actor/critic 3e-4, log sigma [-5,-1], DACER disabled | 2-D bounded actions; +100 sparse success; same profile retained. |
| SAC | Stable-Baselines3 256x2 Gaussian, actor/critic 3e-4, automatic entropy | Standard continuous-action baseline; no PointMaze-specific tuning. |
| SQL | JAX SVGD, 256x2, 16 kernel/value particles, temperature 1, reward scale 1, LR 3e-4 | 2-D bounded actions and sparse reward are accepted; no PointMaze-specific tuning. |
| MEOW | Pinned CleanRL flow, alpha .2, log sigma [-5,-.3], Q LR 1e-3, tau .005 | The same native flow/Q adapter completed the short run; no PointMaze-specific tuning. |
| MFPO | Pinned MeanFlow 256x3, actor/critic LR 3e-4, T2, learned entropy temperature, distributional Q support [-1600,1600] | Support includes the +100 terminal reward. `sample_actions` directly samples its flow in our evaluator; the separate upstream candidate-selection evaluation is not used. No PointMaze-specific tuning. |
| DIPO | Pinned DDiffPG adapter, actor 3e-4, critic 5e-4, C51 Q support [0,120], gamma .99, tau .05, 5 diffusion steps | Support includes the 0/+100 sparse returns; numerical projection guard retained. No PointMaze-specific tuning. |
| TD3 | Stable-Baselines3 256x2, actor/critic 3e-4, exploration noise .1, target noise .2, policy delay 2 | Deterministic policy evaluation, exploration noise only during training; no PointMaze-specific tuning. |

For all seven, replay capacity is 1M, warmup 8192, gamma .99, and the shared
16-updates/256-transitions profile remains. The old PointMaze runs demonstrate
that these configurations execute and produce finite raw rollouts; they do not
prove each baseline is optimally tuned for this task. In particular the methods
have different entropy, architecture and action-selection conventions. Results
must be described as a comparison under this recorded profile, not as a
hyperparameter-matched or paper-exact reproduction. Any subsequent method-wise
tuning belongs in separate named experiments.
