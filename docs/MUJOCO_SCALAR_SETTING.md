# mujoco-setting: scalar-critic ablation of critic-dime-no-anchor

Base: `critic-dime-no-anchor` at `f8f622b09c760e011ef956aa9bdc30cdd336d8e2`.
The default entry now selects `mujoco_setting` (Humanoid-v4). The original
`optiq_dime_no_anchor`, `optiq_dime_mujoco`, and Dog configs remain unchanged.

Reference clarified by the user: [the existing scalar MuJoCo comparison](https://wandb.ai/online-optiflow/online_optiq_sac_td3_sql_h256x3_mujoco5_5seed).
The W&B API returned75 SAC/TD3/SQL runs (no OptiQ runs in this project).
Humanoid seed1 runs `c9ntvgaa` (SAC), `t8tc8ofq` (TD3) and `a5c696vv` (SQL)
all record `tau=0.005`. SAC and TD3 explicitly record256x3 GELU, LR=3e-4,
batch256, replay1M and one gradient step per environment step.
Here `tau` is the critic target-update coefficient, also called Polyak averaging;
it is not an added entropy or policy-temperature parameter. The original
`dev` OptiQ code also uses `target_tau=0.005`, twin-min backup and plain
`optax.adam(lr)` (defaults0.9,0.999).

This matches the requested critic setup, not every benchmark protocol detail:
the no-anchor branch's5K warmup and1M budget remain unchanged, rather than
silently importing the comparison's10K warmup and5M Humanoid budget.

## Changes

- Two independent 256 x 256 x 256 GELU critics, each returning one **raw Q**.
- No batch renormalization, including no dummy BN statistics or computation.
- One update per environment step (UTD=1), instead of DIME's UTD=2.
- Standard Adam betas (0.9,0.999) for both actor and critic, LR=3e-4.
- Target critic with Polyak coefficient0.005 and twin-min TD backup.
- Scalar TD MSE, summed across the two critics, replaces categorical projection,
  cross-entropy and categorical entropy regularization.
- The inherited support endpoints are ignored for scalar Q: no value clipping,
  softmax, or multiplication by `v_min`.
- Actor weighting and Q diagnostic scripts interpret scalar outputs directly.
- A separate output root and default W&B project identify this ablation.

The scalar backup uses the **minimum of target critics**, not DIME's live mean:

```
y = stop_gradient(reward + gamma * (1 - done) * min(Q1_target_next, Q2_target_next))
loss = mean((Q1 - y)^2) + mean((Q2 - y)^2)
target_params = 0.995 * target_params + 0.005 * critic_params
```

The live current-state critic and target next-state critic are evaluated
separately; CrossQ's combined forward pass is disabled. Q-weight temperature
remains0.25 and is unrelated to the critic Polyak coefficient (`alg.tau=0.005`).
The actor's target-copy coefficient remains1.0 (no actor EMA).

## Unchanged settings

- One-step actor: 256x3 GELU; N=16, four random candidates per center (M=64).
- No anchor; stratified KDE proposals; sigma=0.2, clip=0.5 (2.5 sigma).
- Temperature=0.25; fixed density beta=1 (subsequently requested); no adaptive beta.
- Argmax OT distillation, Sinkhorn epsilon=0.05, 30 iterations, pointwise MSE.
- Batch=256, replay=1M, critic/actor warmup=5K, gamma=0.99.
- 1M steps per run, seeds0,1,2 for the five-task protocol (single-entry default0),
  evaluation every5K with10 stochastic episodes and at start,
  diagnostics every5K, checkpoint every50K. No experiments are launched by setup.

## Run

The current five-task protocol is documented in [MUJOCO-README.md](../MUJOCO-README.md).

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

The final GELU / default-Adam / target-min implementation passed21 tests
(39.60s in an isolated CPU-only checkout): all10 scalar tests
and11 retained no-anchor regression tests. The two historical git-source
sampling comparisons were deselected because the temporary checkout has no git
object database. No W&B run or benchmark training was started by validation.

The current suite checks the exact twin-min TD target, Polyak update, terminal masking, stopped target
gradients, scalar values beyond the old categorical support, unchanged actor
configuration, empty BN statistics, checkpoint serialization, and real Ant-v4 /
Humanoid-v4 training for8 environment steps with256x3 networks and6 updates each.
The retained categorical critic training test also passed.

Run the scalar tests without the baseline's automatic online-W&B fixture:

```bash
python -m pytest --noconftest -q tests/test_scalar_critic.py
```
