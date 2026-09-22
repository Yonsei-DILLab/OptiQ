# OptiQ experiment versioning

The user requested on 2026-09-17 that this work use branch `heejoon` and that
experiments be committed before they are launched.

- Commit the exact experiment code, configurations, launch scripts, and protocol
  before new training or a changed setting is launched. Record the full commit
  SHA with the source manifest and submitted jobs.
- An unchanged checkpoint resume retains its original source commit; document
  the resume without creating empty commits per seed or scheduler retry.
- Preserve running/frozen source snapshots. Keep Git provenance in a sidecar
  when adding metadata to already-running experiments, and disclose post-launch
  commits explicitly.
- Do not commit credentials, SSH/W&B keys, checkpoint/sample archives, virtual
  environments, or generated logs. Data remains on dildata. Small existing
  source-manifest fixtures may be retained for reproducibility.
- Do not force-push or overwrite collaborators' changes. Preserve shared branch
  history and the source/configuration used by earlier experiments.

Recent self-contained experiments are under `analysis_tools/studies/`.

User clarification (2026-09-18): MuJoCo environment order is a scheduling
preference, not a completion dependency. Independent runs should be eligible
together and use available GPUs concurrently. Use priority/nice to express
environment preference; add cross-environment `afterok` only when the user
explicitly requests a completion gate. Keep explicitly deferred runs deferred.

User default (2026-09-21): For new Direct GMM/TRG runs, keep actor log_std
bounds fixed at [-5, -1] (log_std_min=-5.0, log_std_max=-1.0). Explicitly
requested ablations may override these bounds in their own experiment profiles;
do not promote ablation bounds to defaults based on their results. Preserve
running/frozen snapshots. The requested Ant/Humanoid 20k cap 0 versus -2
comparison is an explicit experiment-only exception.

User visualization default (2026-09-21): OptiQ GMM figures use μ-only action
outputs with conditional Gaussian noise removed, retaining each run's original
fixed/random latent prior. Near/coverage/MMD labels, tables and curves must use
the same μ-only samples. Keep full-policy σ-noise results as a clearly labeled
supplement, never relabel full-policy metrics as μ-only. Other baselines retain
their native generator outputs and are labeled accordingly. Preserve raw metrics,
training, RL reward evaluation and frozen source. For old frozen campaigns,
render a separate post-hoc report with reporting-source provenance.

User constraint (2026-09-21, latest): Keep GMM40 OptiQ N=M=64 in this
parameter search. Do not increase N/M for the coverage/near goal. Existing
N=M256 results are historical only and do not satisfy the current goal.

User experiment selection (2026-09-22): Future AntMaze comparisons use OptiQ, SAC, DIPO, and MFPO; MEOW runs/queues were cancelled. The requested fresh NovelD10 v1-v4 seed0 1M campaign should evaluate every250k and save the full checkpoint only at final1M. The latest instruction holds new training until saved-checkpoint trajectories have been inspected to assess coefficient10. Preserve cancelled-run logs/checkpoints and do not relaunch older campaigns.

User selection (2026-09-22, latest): v1 stays at NovelD 0.01; add only DIPO
seed0 for 1M interactions, evaluating every250k and saving the full checkpoint
only at final1M. Keep original OptiQ/SAC/MFPO v1 controls. Investigate literature
and existing settings before deciding coefficient studies for v2/v3/v4. Do not
launch the held all-maze coefficient10 campaign, ERIR, xy novelty or new shaping.

User selection (2026-09-22, newest): register fresh v2/v3/v4 OptiQ/SAC/DIPO/MFPO
seed0 1M runs at NovelD0.1 on both four-GPU servers. Existing v1 DIPO0.01 stays
running. Evaluate every250k and save full state only at final1M. Use independent
per-job backfill immediately when a slot opens; no all-method/all-maze barrier.
Keep dense reward, other native hyperparameters and NovelD structure unchanged.

User stop (2026-09-22, supersedes v2-v4 launch instructions above): all
v2/v3/v4 NovelD0.1 running and pending jobs were stopped/cancelled. Only the
already running v1 DIPO NovelD0.01 may continue from its unchanged frozen source.
Do not resume cancelled jobs without a new explicit user instruction.

User source replacement (2026-09-23): replace the custom AntMaze implementation
with the complete official supersglzc/ddiffpg repository under antmaze/. Preserve
the upstream files unchanged, their licenses, and provenance; see
docs/ANTMAZE_UPSTREAM.md and docs/antmaze-ddiffpg-upstream.json. Preserve previous
results and running/frozen source snapshots. This is a source import, not an
experiment launch or authorization to run upstream defaults. OptiQ and MFPO
adapters are absent from this upstream snapshot.

User approval (2026-09-23, latest): stop old v1 DIPO and run fresh official
DDiffPG AntMaze v1-v4 with OptiQ/SAC/DIPO/MFPO, seed0, total1M interactions
each, using8 GPUs. User explicitly chose64 environments per run and batch4096
for all four methods. Use common original DDiffPG update ratio: collect64 then
2 learner updates. Warmup8192 is included in1M. See antmaze_experiments/PROTOCOL.md.
Keep upstream antmaze/ files unchanged; place integration outside that tree.
Validate runtime, real batch4096 updates and accounting before main launches.

User correction (2026-09-23, supersedes sparse/1M profile): use dense negative
nearest-goal Euclidean distance without sparse goal bonus. Respect upstream
per-maze budgets: v1/v2 3M, v3 4M, v4 5M. Keep all four methods at64 envs,
batch4096, collect64/update2, seed0, NovelD.01, evaluation250k, final-only full
checkpoint. Cancel sparse campaign, preserve logs, and restart fresh dense jobs.
Dense DIPO requires negative critic support; document this compatibility override.
MaxEntDP uses dense distance reward and reports1M trajectories, but its public
source does not establish an AntMaze NovelD setting. Do not describe .01 or
our parallel/batch/update profile as MaxEntDP defaults.
