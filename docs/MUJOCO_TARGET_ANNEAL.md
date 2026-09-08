# Ant and HalfCheetah target temperature annealing

Run fresh Ant-v4 and HalfCheetah-v4 experiments, each with seed 0, using the
no-anchor OptiQ actor and shared JAX DIME training path. The configuration
inherits `optiq_dime_reach_target_anneal`, including its exact target schedule:

```text
T(t) = 10 * (0.25 / 10) ** min(t / 400000, 1)
candidate weights = softmax(Q / T(t) - 0.1 * log(g))
```

The clock counts environment steps from zero, including the 5,000-step
random-action warmup. Temperature reaches 0.25 at 400,000 steps and remains
there through 1,000,000 steps. At 80k intervals it is approximately
10, 4.7818, 2.2865, 1.0934, 0.5228, 0.25. Only the target temperature anneals.

The KDE remains at std 0.2 and clip radius 0.5, with 16 centers, four random
samples per center, no inserted anchors, and a 16 x 64 transport plan.
Density beta stays 0.1, Sinkhorn epsilon 0.05 with 30 iterations, argmax
transport targets, pointwise MSE, TD noise std 0.2 / clip 0.5, UTD 2,
batch size 256, and the two 2048 x 2048 DIME critics. All settings are inherited
from the Reach protocol except the environment, output root, and the previously
approved Gym critic support **[-1600, 1600]** in place of Myo's [-3600, 3600].

Evaluation runs every 5,000 steps with 10 stochastic episodes, including the
initial evaluation. Diagnostics are logged every 5,000 steps and checkpoints
every 50,000 steps, plus the first training step. These are fresh runs and do
not load the old Ant checkpoints, which used inserted anchors and narrower KDE.

```bash
bash scripts/run_mujoco_target_anneal.sh --list
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mujoco_target_anneal.sh --task 0
CUDA_VISIBLE_DEVICES=1 bash scripts/run_mujoco_target_anneal.sh --task 1
```

Use `/workspace/.venv-optiq-no-anchor` and the ignored `.env`. The supervisor
template is `scripts/supervisor_mujoco_target.conf.example`; jobs do not restart
automatically. GPUs 0/1 replace the two object-hold baseline workers. The Reach
target-temperature experiments on GPUs 2/3 continue in their existing worktree.

Results are written to `outputs/optiq_dime_mujoco_target_anneal` under unique run
directories. W&B uses the existing `optiq_dime_no_anchor` project unless the
environment overrides it, with separate environment-specific target-temperature
groups. Logs, evaluations, configuration, and previously saved checkpoints from
the stopped object-hold workers remain in their original directories. A stop
does not guarantee a new checkpoint at the exact interruption step.
