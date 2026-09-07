# mujoco-setting: scalar-critic ablation of critic-dime-no-anchor

Base: `critic-dime-no-anchor` at `f8f622b09c760e011ef956aa9bdc30cdd336d8e2`.
The default entry now selects `mujoco_setting` (Humanoid-v4). The original
`optiq_dime_no_anchor`, `optiq_dime_mujoco`, and Dog configs remain unchanged.

## Changes

- Two independent 256 x 256 x 256 ReLU critics, each returning one **raw Q**.
- No batch renormalization, including no dummy BN statistics or computation.
- One update per environment step (UTD=1), instead of DIME's UTD=2.
- Scalar TD MSE, summed across the two critics, replaces categorical projection,
  cross-entropy and categorical entropy regularization.
- The inherited support endpoints are ignored for scalar Q: no value clipping,
  softmax, or multiplication by `v_min`.
- Actor weighting and Q diagnostic scripts interpret scalar outputs directly.
- A separate output root and default W&B project identify this ablation.

The backup deliberately preserves DIME's **mean**, not the minimum:

```
y = stop_gradient(reward + gamma * (1 - done) * mean(Q1_next, Q2_next))
loss = mean((Q1 - y)^2) + mean((Q2 - y)^2)
```

The combined current/next-state forward pass is retained, but without batch
normalization it does not couple the samples via batch statistics. The inherited
live-network bootstrap is retained (not switched to a separate target critic).
This is a scalar-critic ablation, **not** a reproduction of the older
256x3 scalar experiment's entire target-network/optimizer/normalization protocol.

## Unchanged settings

- One-step actor: 256x3 GELU; N=16, four random candidates per center (M=64).
- No anchor; stratified KDE proposals; sigma=0.2, clip=0.5 (2.5 sigma).
- Temperature=0.25; fixed density beta=0.1; no adaptive-beta modification.
- Argmax OT distillation, Sinkhorn epsilon=0.05, 30 iterations, pointwise MSE.
- Batch=256, replay=1M, critic/actor warmup=5K, gamma=0.99.
- Adam settings are inherited; LR=3e-4 for actor and critic.
- 1M steps, seed=0, evaluation every5K with10 stochastic episodes and at start,
  diagnostics every5K, checkpoint every50K. No experiments are launched by setup.

## Run

Use the existing isolated environment and credential setup described in
[NO_ANCHOR_BASELINE.md](NO_ANCHOR_BASELINE.md).

```bash
python run_optiq_dime.py --config-name=mujoco_setting benchmark=humanoid seed=0
python run_optiq_dime.py --config-name=mujoco_setting benchmark=ant seed=0
```

For a fixed-beta comparison, override the existing canonical alias:

```bash
python run_optiq_dime.py --config-name=mujoco_setting benchmark=humanoid \
  alg.actor.density_correction_beta=0.001
```

`WANDB_PROJECT`, if set, still takes precedence over the new project default.
The original DIME baseline is explicitly available with
`--config-name=optiq_dime_no_anchor benchmark=humanoid`.

## Validation (2026-09-07)

In an isolated CPU-only checkout, 21 tests passed (39.51s): all10 scalar tests
and11 retained no-anchor regression tests. The two historical git-source
sampling comparisons were deselected because the temporary checkout has no git
object database. No W&B run or benchmark training was started by validation.

Checks include the exact twin-mean TD target, terminal masking, stopped target
gradients, scalar values beyond the old categorical support, unchanged actor
configuration, empty BN statistics, checkpoint serialization, and real Ant-v4 /
Humanoid-v4 training for8 environment steps with256x3 networks and6 updates each.
The retained categorical critic training test also passed.

Run the scalar tests without the baseline's automatic online-W&B fixture:

```bash
python -m pytest --noconftest -q tests/test_scalar_critic.py
```
