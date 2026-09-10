# Humanoid v2 calibration and launch decision

Selected on 2026-09-10 UTC, before the four production runs. Production uses
**T=0.5 fixed** in both the soft TD target and teacher Boltzmann weights,
**beta=1**, pre-tanh teacher KDE **h=0.8**, initial conditional actor **sigma=0.5**,
log-std bounds **[-5,1]**, entropy **M=16**, **16x64** OT, raw squared-action cost,
Sinkhorn **epsilon=0.25 / 100 iterations**. No anchors, hard proposal cutoff,
TD smoothing, extra uniform rollout exploration, or added actor entropy/Q loss.
The common 5k random warmup is retained.

## Temperature evidence

All four pilots started from scratch with seed 0, full 256x3 actor/twin-critic
networks, 20k environment steps, and zero extra uniform exploration. The first
5k steps were common warmup. The rows below are averages of recorded metrics
from steps 15k through 20k; ESS has maximum 64. Entropy and sigma use normalized
action coordinates. `-T log g_M` is the entropy term before discounting and
terminal masking; a negative differential entropy is valid.

| T | ESS / 64 | Entropy lower estimate (nat) | Mean -T log g_M | Mean reward | Mean sigma | 20k eval return |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 8.65 | -28.90 | -2.89 | 5.24 | 0.114 | 446.4 |
| 0.25 | 21.79 | -12.99 | -3.25 | 5.21 | 0.191 | 381.4 |
| **0.5** | **31.06** | **-7.76** | **-3.88** | **5.18** | **0.215** | **395.0** |
| 1.0 | 34.63 | -4.50 | -4.50 | 5.21 | 0.224 | 466.8 |

T=0.5 substantially improves average importance ESS over T=0.25. The additional
ESS increase at T=1 is smaller, while the entropy term grows relative to the
roughly 5.2 immediate reward. The lower log-std clamp was reached by 13.5%, 11.0%,
4.9%, and 0.6% of probed component/action coordinates, respectively. T=0.5 is a
compromise between weight concentration and the entropy contribution to the
backup, not a demonstrated return optimum. Some states still have concentrated
weights: the mean per-batch minimum ESS at T=0.5 is 1.90. No adaptive ESS rule is
introduced. Sigma bounds, entropy, saturation and ESS remain logged for review.

Evaluation used only three stochastic episodes per checkpoint. These 20k-step,
one-seed pilots cannot establish long-run performance or a statistically reliable
ranking. Production uses the original ten-episode evaluation protocol.

W&B records (all in `OptiQ/optiq_mujoco_v2_calibration`):

- [T=0.1](https://wandb.ai/OptiQ/optiq_mujoco_v2_calibration/runs/0fqj410k)
- [T=0.25](https://wandb.ai/OptiQ/optiq_mujoco_v2_calibration/runs/1af5lpx7)
- [T=0.5](https://wandb.ai/OptiQ/optiq_mujoco_v2_calibration/runs/o4f36giz)
- [T=1](https://wandb.ai/OptiQ/optiq_mujoco_v2_calibration/runs/pl45mfar)

Earlier pilots with behavior_uniform_probability=0.1 are preserved as superseded
records and excluded from this selection. All production seeds restart at step 0.

## Entropy resolution

On each final actor's saved 128-state probe, eight repetitions evaluated the same
generated action with M=1/4/16/64 conditional components. Every estimate includes
its generating component. The paired M=64 minus M=16 entropy differences were
0.249, 0.108, **0.037**, and 0.00066 nat for T=0.1/0.25/0.5/1.
At the selected T=0.5, the change in the undiscounted entropy contribution is
about **0.019 reward units**, so M=16 is retained. This is a finite-resolution
sensitivity check on early actors, not an estimate of error against exact policy
entropy. The true marginal is still unavailable.

## Initial proposal and transport checks

128 states collected during a separate random Humanoid rollout were used to
compare initial sigma {.3,.5,1}, h {.2,.4,.6,.8,1}, and raw-cost epsilon
{.05,.1,.25}. With controlled sigma=.5, h=.2 gave ESS 2.98/64, while h=.8 gave
18.36/64. Initial estimated policy entropy was 8.40 nat. Student coordinate
saturation at abs(action)>.99 was zero, and teacher saturation was about 0.55%.
The teacher bandwidth is measured before tanh; it is not the learned actor sigma.

Epsilon=.25 with 100 Sinkhorn iterations uses the pseudocode's raw squared
action-distance cost. It is a separate regularization parameter from T=.5.
The T=.5 pilot's late mean absolute row-marginal error was 1.8e-8.

Random initial log-std head outputs on unnormalized Humanoid observations could
hit their clamps. Initializing its kernel to zero and bias to log(.5) provides
controlled initial entropy; both output heads remain trainable. The mu head uses
variance-scaling initialization 1e-4.

## Reproducibility and validation

Local machine-readable evidence is under `outputs/v2_validation/`:
`initial_calibration.json`, `pilot_summary.json`, `pilot_launch.json`, and
`final_tests.log`. Pilot configs, logs, evaluation arrays and 20k checkpoints are
under `outputs/v2_calibration/`. The scripts for both analyses are in `scripts/`.

The common-path test suite passed **94 tests** before final temperature selection.
It covers density/Jacobian mathematics, state isolation, generating-component
inclusion, KDE sampling, scalar and categorical entropy backups, terminal and
time-limit handling, both actor heads' gradients, stopped Q gradients, actual
Humanoid training/checkpoint restore, and legacy path regressions. All four GPU
pilots completed 20k steps without non-finite logged metrics. The final T=.5
configuration passed all **17 v2 tests** again; the integration check reads
emitted CSV records to verify that the backup entropy coefficient equals T.

Production: Humanoid-v4 seeds 0/1/2/3, one worker per GPU, 1M environment steps
each, under `OptiQ/optiq_mujoco_v2_scalar_h256x3_no_anchor_idac_4seed_1m`.
