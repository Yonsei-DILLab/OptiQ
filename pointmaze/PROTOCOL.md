# Multi-goal PointMaze: completed T=5 profile

This is a separate benchmark configuration, not a change to MuJoCo or GMM40.
The completed experiments used one training seed (0) per map; they are not
five-seed results. No new full training is launched by adding this package.

| Setting | Value |
| --- | --- |
| Fixed temperature / density correction | 5 / beta=1 |
| Actor / scalar twin critics | 256 x 2, GELU |
| Actor / critic optimizer | Adam, learning rate 3e-4, betas (0.9, 0.999) |
| Discount / target update coefficient | 0.99 / 0.005 |
| Replay capacity / batch size | 1,000,000 / 4096 |
| Parallel collectors | 256 |
| Collection / optimization | 256 transitions, then 16 independent batch updates |
| UTD | 0.0625 |
| Uniform-action warmup | 8192 transitions, included in the budget |
| Total interactions / learner updates | 1,000,192 / 62,000 |
| Latents / components / candidates | fresh N(0,I_2), N=M=64 |
| Conditional distribution | diagonal box-truncated Gaussian on [-1,1]^2 |
| log sigma bounds / initial | [-5,-1] / -1 |
| Mean-head initialization variance scale | 1e-4 |
| Proposal sigma floor | exp(-5) |
| Actor objective | direct value-weighted marginal mixture NLL |
| Gradient clipping / DACER / NovelD / entropy backup | all disabled |

The warmup lasts 32 collector batches. The remaining 3875 batches each
perform 16 updates: `(1,000,192 - 8192) / 256 * 16 = 62,000`.
The 256 environments are independent, synchronously stepped CPU simulators;
this does not claim GPU-vectorized MuJoCo physics.

## Environment

The unchanged DrAC multi-goal PointMaze implementation is included under
`drac_upstream/`. Observations are (x,y,vx,vy), actions lie in [-1,1]^2.
The original sparse reward is +100 upon reaching any goal and zero otherwise.
Keep native resets, collision behavior, success termination and time limits.

| Map | Goals | Episode horizon |
| --- | --- | --- |
| Simple | 4 | 150 |
| Medium | 4 | 300 |
| Hard | 8 | 600 |

Light-gray map cells are passable during training. They become obstacles
only in the separate final robustness evaluation. Training uses no additional
reward shaping, entropy reward, expert data, or DrAC diversity objective.

The original training adapter stores termination OR truncation as the TD
terminal mask, including horizon endings. This release intentionally preserves
that convention rather than silently switching to timeout bootstrapping.
Replay stores the actual executed clipped action and pre-reset next observation.

## Running

From the repository root, after installing `requirements.txt`:

```bash
python -m pointmaze.run --maze simple --output outputs/pointmaze-simple-T5-s0
python -m pointmaze.run --maze medium --output outputs/pointmaze-medium-T5-s0
python -m pointmaze.run --maze hard --output outputs/pointmaze-hard-T5-s0
```

Each command is one independent training run. No server, scheduler, credentials
or W&B account is required. Smaller command-line budgets are smoke tests, not
the completed paper profile. Existing output directories are never overwritten.

## Evaluation and release boundary

Training collection and TD actions retain conditional Gaussian sampling.
Evaluation draws a fresh normal z for each action and uses the bounded mean
only: no conditional sigma noise, external exploration noise, or zero-z action.
Evaluate every approximately 200k transitions (rounded up to a collector batch)
with 200 episodes and finally 500 episodes, plus final obstacle evaluation.
Save raw paths (including failures), per-episode goals/returns and policy/critic
states. Evaluation restores the training policy RNG afterwards.

This minimal release uses serial evaluation with explicit seeds; it does not
claim bitwise replication of the historical batched evaluation streams.
It reports single-episode success and per-goal counts, not the paper's
five-trial robustness metric. Raw rollout arrays permit separate analysis.
Policy checkpoints support inspection, not exact interrupted-training resume:
the replay, simulator and all training RNG states are not serialized here.
The training implementation reuses the cleaned iBOLT learner; numerical parity
and validation limitations from `VALIDATION.md` still apply.
