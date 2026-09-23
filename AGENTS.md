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

User reversal (2026-09-23, newest): stop dense campaign and restart all16
OptiQ/SAC/DIPO/MFPO v1-v4 seed0 policies using original DDiffPG sparse reward
and NovelD.01, with256 parallel environments. Check official repo and actual
environment defaults before launch. SAC/DIPO native batch4096/update_times8/
warm_up32/replay1M are the shared data/update profile; preserve native models/LRs.
Restore DIPO support[0,5]. Original budgets v1/v2 3M,v3 4M,v4 5M; original global
counter excludes warmup and stops strictly above max_step. Retain user's
eval250k/final-only full save. Independent backfill on8 GPUs, no completion
barriers. Cancelled dense/sparse64/older queues stay cancelled, data preserved.
Upstream159 files remain unchanged; see antmaze_experiments/PROTOCOL.md for
verified environment settings and adapter/runtime/evaluation exceptions.

User approval (2026-09-23, latest): stabilize DIPO numerical projection/BCE and gradients outside vendored antmaze/. Use dense negative nearest-goal distance and NovelD OFF. First inspect 10k-learner-update v1/v3 trajectories for OptiQ/SAC/DIPO/MFPO, then fresh 16-policy parallel native-budget training on8GPUs. Keep256env,batch4096,8updates/256transitions,other native hyperparameters and original physics. DIPO negative support is an explicit dense compatibility override. See antmaze_experiments/DENSE_OFF_PROTOCOL.md. Cancelled/failed older campaigns remain inactive.

User evaluation preference (2026-09-23, latest): primary AntMaze trajectory plots must use sampled random starting positions, not one arbitrarily fixed start. New main launch profile uses evaluation-only xy uniform[-2,2] for all four mazes, keeping native training reset settings. Record this explicit evaluation override for v2-v4 and never relabel completed fixed-start data. Preserve fixed-start results as supplementary only. Dense+NovelD-off 10k-update probes are completed; the main native-budget queue has not been registered while the user discusses sparse rewards and exploration design.

User launch approval (2026-09-23, newest): proceed now with all16 dense-reward
AntMaze policies, OptiQ/SAC/DIPO/MFPO x v1-v4, seed0. This resolves and supersedes
the preceding main-queue hold. Use the already validated dense + NovelD OFF
profile in DENSE_OFF_PROTOCOL.md, original 3M/3M/4M/5M native budgets,256envs,
batch4096,8updates per256transitions,8GPUs with independent2second backfill.
Keep validated numerical safeguards and randomized evaluation starts. Preserve
all older cancelled queues and completed probe results; do not relaunch them.

User restart and correction (2026-09-23, latest): stop all dense jobs and cancel
their pending queue. Launch only OptiQ v1/v2/v3/v4, seed0 each (four total),
using original sparse reward + NovelD0.01. The user explicitly withdrew removal
of the sigma cap: retain log sigma[-5,-1],initial-1 and set temperature0.01.
Use250k evaluations with40 episodes per native/direct mode, save intermediate
evaluation-only policy checkpoints, and retain final100 episodes/full-state save.
Preserve256env,batch4096,8updates/256transitions and native3M/3M/4M/5M budgets.
The four jobs use server180; server199 receives source only. Do not launch
baselines/extra seeds or resume cancelled campaigns. See SPARSE_T001_PROTOCOL.md.

User logging separation (2026-09-23, latest): all AntMaze runs belong to
W&B OptiQ/antmaze, including historical runs moved from OptiQ/gmm-trg. Use
antmaze_experiments/settings.py logging constants for new online and offline
uploads. Preserve run IDs, metrics, local logs, frozen source and checkpoints;
record relocated URLs in separate metadata. Do not restart training to change
the project. Other benchmarks keep their existing projects. See
antmaze_experiments/WANDB_PROJECT.md.


User LR/architecture approval (2026-09-23, newest): replace the previous OptiQ
AntMaze run with fresh v1-v4 seed0 jobs using DDiffPG main-method network LRs
(actor3e-4,critic5e-4), and use256x3 for BOTH OptiQ actor and twin critics in
AntMaze. RND remains the NovelD estimator at its existing1e-4 LR. Preserve
T=.01,sparse+NovelD.01,log sigma[-5,-1],tau.005,Adam,batch4096,256env and the
8/256 update ratio, other hyperparameters, original budgets and40-episode
intermediate/random-start evaluation. Do not adopt DDiffPG action_lr.03 or
change optimizer family/tau. The previous four jobs were already completed
when checked; preserve their results/frozen source and do not relaunch them.
Launch only four new OptiQ jobs on180, share committed source with199, and log
toOptiQ/antmaze. See antmaze_experiments/DDIFFPG_LR_256X3_PROTOCOL.md.


User launch approval (2026-09-24, newest): run sixteen fresh AntMaze policies,
OptiQ/SAC/DIPO/MFPO x v1-v4, seed0, using dense negative nearest-goal distance
and NovelD OFF. Preserve latest OptiQ256x3 actor/critic,T=.01,actor3e-4/critic5e-4,
log sigma[-5,-1],DACER on and other settings. Keep native baseline models/LRs,
validated DIPO numerical guards and dense support[-6000,5]. Use256env,batch4096,
8updates/256transitions,native3M/3M/4M/5M budgets,40-episode random-start
intermediate evals,final100 and existing OptiQ policy checkpoints. Register a
new16-job campaign on8GPUs with independent2second backfill; old cancelled
queues remain cancelled. W&B OptiQ/antmaze. Commit/push/share frozen source
before launch. See antmaze_experiments/DENSE_OFF_16_CURRENT_PROTOCOL.md.


User temperature experiment (2026-09-24, newest): stop currently running SAC
jobs and launch four fresh OptiQ policies, v1-v4 seed0, dense reward and NovelD
OFF, temperature1. Preserve the latest OptiQ256x3 actor/critic and native
3M/3M/4M/5M budgets,256env,batch4096,8updates/256transitions, all other settings
and40-episode intermediate/random-start evaluation. Preserve existing DIPO
training and completed results; do not restart held MFPO/cancelled SAC queues.
Use direct-gmm-trg-antmaze, W&B OptiQ/antmaze, committed/shared frozen source.
See antmaze_experiments/DENSE_T1_PROTOCOL.md.


User follow-up queue (2026-09-24, newest): queue twelve OptiQ AntMaze
annealing runs (v1-v4 x10→1,10→.25,10→.5,seed0),dense reward/NovelD OFF.
Exclude8192 warmup; linearly anneal over the next1M environment transitions,
then hold the final temperature through existing3M/3M/4M/5M budgets. After
these queue entries, schedule missing dense baselines: MFPOv1-v4 and SACv4.
This explicitly authorizes these baseline replacements despite prior holds.
SACv2 final result/checkpoint is complete despite controller cancellation; do
not duplicate it. Preserve current T1/DIPO jobs and every old source/result.
Use independent per-slot backfill, no cross-maze barrier, and W&B OptiQ/antmaze.
See antmaze_experiments/ANNEAL_BASELINES_PROTOCOL.md.
