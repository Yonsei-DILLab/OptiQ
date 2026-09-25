# Online four-goal navigation comparison

The comparison uses seed 0 for each of SAC, JAX SVGD SQL, DIPO, MFPO and
Direct GMM OptiQ (iBOLT) in **both** environments. A single policy is learned
per method and task. Goals reached by different training seeds must never be
pooled to imply a single multimodal policy.

## Tasks

- `4way`: the **existing wall-free** task previously trained for 10k/20k,
  as the user explicitly selected after inspecting the walled ICML layout.
  State/action are both 2D, with central reset and goals `(±5,0),(0,±5)`.
  Reward is `−30||a||² − min_goal ||s'−g||² + 10 on success`, identical in
  all four directions. Episode limit 20. Train for **100,000 environment
  transitions**. The author's different, walled ICML task stays in `4way/`
  as source provenance only and is not used in these runs.
- `pointmaze`: official installed Farama `PointMazeEnv` MuJoCo physics with a
  four-goal cross map and one fixed central reset cell. The sampled native
  single-goal target is hidden and ignored. Reward is exp(−distance to the
  nearest of the four goals), success radius 0.45, episode limit 300. This
  task mapping and reward are a disclosed adaptation, not an official ID.

The wall-free 4way policies see `(x,y)` and choose a displacement in `[−1,1]²`;
PointMaze policies see `(x,y,vx,vy)` and choose a continuous force in `[−1,1]²`.
The five algorithms use the same task reward, reset distribution, evaluation
states, environment-transition budget, replay sampling and UTD within a task.
Policy rollouts use the directly sampled action (including conditional noise
for OptiQ); OptiQ μ-only random-latent rollouts are supplementary.

The first wall-free 4-Way profile uses 16 simultaneous collectors, batch 256,
and 16 learner updates per collection of 16 transitions: UTD=1, as in the
successful 20k OptiQ diagnostic. Warmup is 1,024 transitions, included in
the 100k budget. Evaluate 100 rollouts at each 20k transition checkpoint
and save the policy/critic together with the independent policy/Q grid probe.
Run each method in a separate process. Results from this profile must not be
mixed with the author's distinct walled ICML environment.

After the T=3, 100k OptiQ run reached all goals but only 8/100 west-goal
episodes, run the user-selected temperature grid T=1,3,5,10 with the same
seed and 100k budget. T=3 is already complete; add only T=1,5,10. Preserve
the T=3 checkpoint and both policy/μ-only measurements.
Select a temperature by the weakest goal count among 100 direct-policy
rollouts, subject to a high success rate, rather than hiding a rare goal in
aggregate success. No environment, reward, UTD, network, or sigma setting
changes in this comparison.

## Longer T=1 critic check

The 100k comparison showed all four goals under T=1, but the user also wants
the learned Q surface to converge for the paper-style figure. Run a separate,
fresh seed-0 OptiQ T=1 policy for 500k environment transitions using
`launch_4way_t1_500k.sh`. This is **not** a continuation of the 100k run:
the saved 100k checkpoint does not contain the full environment and RNG state.
Retain the same wall-free task, 16 collectors, batch 256, UTD 1, 1,024 warmup,
256×2 networks, N=M=64, log sigma [−5,−1], DACER off, reward, and optimizer.
At each 50k transitions, retain 100 direct-policy and 100 mu-only center-start
trajectories, the policy/Q grid probe and an evaluation-only checkpoint; save
replay with the final checkpoint. Inspect four-goal reach **and** the learned
Q surface over time. A smoother surface alone is not proof of value accuracy;
compare its goal ranking and sampled-policy returns before claiming that.

## Visualization

At each checkpoint, retain 100 independent center-start sampled-policy
rollouts, four goal hit counts, returns and raw `(x,y)` tracks. The 4-Way
paper-style panel has two rows for each method:

1. The same four-Gaussian display contour from the official DACER
   `relax_env/multigoal.py::_plot_position_cost` (commit `9f22f29`; σ=1.7,
   amplitude `40/(2πσ²)`, Gaussian contributions summed) for every method,
   overlaid with independent red policy-action samples at the same valid
   states. Arrow length is proportional to action magnitude. This background
   is a task-geometry reference, **not** policy density or the full reward
   (which also penalizes action energy).
2. The method's own `E_{a∼π}[Q((x,y),a)]` surface. Each critic's
   objective/aggregation is labeled. These
   are learned estimates, not ground-truth returns; absolute scales need not
   agree across algorithms.

The top panels also report four goal hits and failures. For OptiQ, retain a
separate μ-only panel so conditional Gaussian noise cannot be mistaken for
random-latent behavior. A trajectory-only figure is retained alongside the
two-row representation figure for unambiguous goal/path inspection.

The official DACER code uses a Gaussian reset with initial position σ=0.1
and registers its task with a 100-step limit. Those settings differ from the
user-selected earlier wall-free 4-Way diagnostic (fixed origin, 20-step limit)
and are **not** silently applied to the already running 100k experiment.

## Provenance and launch gate

Commit exact source, config, launch and reporting scripts before any training.
Use an immutable source SHA per run and independent GPU jobs with backfill.
Do not displace already running unrelated GPU jobs. On completion, verify
transition/update counters, source/config, per-method checkpoints and raw
rollouts before cross-method comparison. The 30-minute scheduled watcher
checks live process handles, controller status, errors, eval coverage and
four-goal OptiQ reach; it does not relaunch solely because a status read times
out. If iBOLT collapses, preserve evidence, stop that run, diagnose, commit
the changed setting and launch a new identifiable run.
