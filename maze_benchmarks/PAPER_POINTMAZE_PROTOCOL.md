# Multi-goal PointMaze trajectory and robustness comparison

This is a new campaign. It does not alter or restart the older cross-shaped
`pointmaze` task or the wall-free 4/8/12/16/32-way campaigns.

## Source and tasks

The maze physics, maps, rewards and collision behavior are the unmodified
`PneuC/DrAC` sources at commit `4e718983ea29aa3a856955f553a99795fcb4e94d`,
copied under `pointmaze/drac_upstream/` with source attribution. Our thin
`pointmaze/drac_paper.py` adapter exposes zero-based goal IDs. Simple/medium/
hard contain 4/4/8 goals, have fixed center starts, and terminate by 150/300/
600 steps. The paper's sparse reward is +100 on reaching any goal, zero
otherwise. Cells coded `2` in the source maps are empty during training and
become walls only during obstacle evaluation. The old custom cross task is
unchanged and **must not** be pooled with these results.

Each of OptiQ, SAC, JAX-SVGD SQL, MEOW, MFPO, DIPO and TD3 learns one policy
per maze at seed 0. The paper's environment-transition budgets are
simple 100k, medium 200k and hard 300k. Actual counts round up to one whole
vector collection and are recorded in the run config. This initial single-seed
comparison is not a five-seed statistical reproduction of the paper.

## Fast common collection profile

Try 256 independent MuJoCo environments per policy. On proven out-of-memory,
try 128 then 64; only after that reduce the batch from 4096 to 2048, 1024,
then 256. Every attempted setting has its own preserved real-update preflight.
The selected vector count, batch and update count are recorded for that policy.
No candidate is silently changed mid-training. All policies use the same
environment source, reward, replay size 1M, discount .99, 8192-transition
warmup and per-transition replay sample budget of 256. Thus for 256 envs and
batch 4096, the learner takes 16 updates per 256 fresh transitions. This is
**not** the paper's one batch-256 optimizer step per transition: batch grouping
reduces optimizer calls and changes learning dynamics. Compare actual
environment and optimizer counts; do not label it an exact DrAC replication.
The method-specific policy architectures and optimizers follow the repository's
existing baseline adapters. OptiQ Direct GMM/TRG uses T=3, DACER off,
N=M64, random latent, 256x2, mean init 1e-4 and log sigma [-5,-1].
SQL retains its baseline T=1; the other methods retain their own native
entropy settings. The T=3 PointMaze queue (`paper-pointmaze-seven-t3-20260926`)
replaces an unstarted T=1 queue. `PAPER_POINTMAZE_T3_PLAN.json` fixes host shards;
the original registration and its zero-run cancellation remain archived.
The supplemental `PAPER_POINTMAZE_4090_TRANSFER.json` records a later
schedule-only migration of pending jobs to four idle RTX 4090 GPUs after a
real 256-env/4096-batch preflight. The frozen learning source and method
settings remain unchanged. A transfer is allowed only while each named job is
pending under the server queue lock; the source queue marks it `transferred`
and preserves a sidecar containing the previous queue state. No running or
completed job is migrated. The collector verifies distinct completed names
across all three hosts.
On `vast1`, two DIPO preflights failed before learning because `mujoco_py`
could not locate MuJoCo 2.1 via the supervisor environment. The failed job
records and logs remain intact. `PAPER_POINTMAZE_4090_DIPO_ENVFIX.json`
documents the environment-only correction, its successful independent real
update smoke test, and the separate queue for the three DIPO policies. The
original main queue continues the non-DIPO work. Reporting combines both
queues while retaining the failed-attempt audits and the frozen training SHA.
Before the 4090 repair queue claimed any DIPO jobs, unrelated GPU processes
occupied its two guarded slots. `PAPER_POINTMAZE_DIPO_TO_180.json` records a
second schedule-only migration of all three still-pending DIPO runs to an idle
5090 slot. The 4090 repair queue marks them transferred and keeps its audit;
the 5090 uses the same frozen source and verified 256-env/4096-batch profile.
The exact environment-corrected `vast1` supervisor launch configurations are
in `maze_benchmarks/supervisor/`; these were committed after the initial
4090 launch, and are used only for new workers after the two already-running
OptiQ jobs exit. Their earlier launch configuration remains in supervisor
logs and the original scheduler manifest.

The server-local queue has no method/maze completion gate. Any worker whose
GPU lock becomes available claims the next pending job under a file lock.
Failures pause new claims; running workers finish or surface their own failure.
No failed job is automatically retried. Frozen code is committed, SHA256
manifested, shared with both servers and pinned in every job before preflight.

## Evaluation and figures

Every 20% of the environment budget (plus final), evaluate directly sampled
policy actions from the original fixed center start. Intermediate evaluation
has 200 independent episodes per normal/obstacle setting; final has 500 each.
Keep all raw `(x,y)` trajectories, returns, lengths and goal IDs. OptiQ
random-latent μ-only trajectories are a separate supplement; the direct-policy
plots include conditional Gaussian noise. Figure panels and metrics use the
same direct-policy raw samples for every method.
For each completed OptiQ maze, an additional two-panel figure compares the
direct policy with random-z μ-only from the same checkpoint. Each panel prints
all-episode success and per-goal counts; the visible tracks are the first 100.

- Success rate: fraction of direct-policy episodes that reach any goal.
- Reachable goals: number of distinct reached goals in that policy's episodes.
- Removal SR5: following the original DrAC evaluation, average probability
  that at least one of five independent policy episodes hits a surviving goal
  when a uniformly chosen half of the goals is removed. The combinatorial
  expectation is computed from each five-episode group; no retraining occurs.
- Obstacle SR5: fraction of five-episode groups with at least one success on
  the original map with the premarked latent obstacle cells made solid. The
  actor receives the same observation; no test-time adaptation occurs.

Figures are generated automatically at each checkpoint: base and obstacle
trajectories over the **actual original map**. The report tool renders all
seven methods side by side for each maze and a 4×3 learning-curve grid for
success, reachable goals, removal SR5 and obstacle SR5. Seed 0 alone has no
confidence band. Goal counts, raw rollouts, actual reset states, selected
vector/batch profile and preflight/final checkpoint proofs remain inspectable.
An additional two-row seven-column plate stacks the medium and hard mazes,
matching the layout of the paper's Figures 10 and 11 while labeling every
panel with its own success and reachable-goal counts.

Reference: [Wang, Liu & Pan, *Learning Intractable Multimodal Policies with
Reparameterization and Diversity Regularization*, §5.1 and Appendix A.2](https://arxiv.org/abs/2511.01374).

The separate collector (`python -m maze_benchmarks.collect_pointmaze
--output artifacts/paper_pointmaze_seven_t3_20260926 --source-commit <frozen SHA>`)
downloads completed runs,
checks raw goal counts, checkpoint SHA256 and replay archive integrity, then
renders the combined seven-method figures. Its source revision is recorded
separately from the frozen learning revision.
