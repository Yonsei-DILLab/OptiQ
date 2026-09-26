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
with 200 episodes and finally 500 episodes. Both ordinary and obstacle
evaluations run at EVERY evaluation checkpoint, matching the historical runner.
Save raw paths (including failures), per-episode goals/returns and policy/critic
states. Evaluation restores the training policy RNG afterwards.

Evaluation uses batches of at most128 episodes, retaining finished rows in
the policy batch but never stepping their simulators again. Per-checkpoint
seeds are training_seed+17000+step (ordinary) and training_seed+27000+step
(obstacle); each128-episode chunk adds its episode offset. Collector resets
and policy RNG initialization match the original evaluator. Fresh z is drawn
per action, not held for an episode.

Report single-episode success, reachable goals, per-goal counts, failures,
mean return/length, and five-trial robustness. Group consecutive episode IDs
in groups of five, including failures. Obstacle robustness is the fraction
of groups containing any success. Removal robustness is the exact probability
that at least one reached goal remains when half the goals are uniformly
removed. It uses the corrected original subset scorer, not its historical
half-goals boundary bug. For four goals and two distinct reached goals this
score is5/6, not1. No environment is retrained for removal scoring.
Raw archives use the original xy/returns/lengths/goal_ids/mode/observation_dim/
obstacle schema. The main JSON record separates mu_only and obstacle_mu_only.
Policy checkpoints support inspection, not exact interrupted-training resume:
the replay, simulator and all training RNG states are not serialized here.
The training implementation reuses the cleaned iBOLT learner. Short execution
checks do not establish bitwise equality with historical full-training runs.
