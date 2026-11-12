# Queued Adroit native baselines

Continuation on server 1, explicitly requested after the active iBOLT/SAC
campaign. Execution order is **all 40 iBOLT/SAC runs → DIPO → MEOW → MFPO**.
Each added method has 4 tasks × seeds 0–4 = 20 full runs, so 60 full runs total.
No running learner or its immutable source snapshot is modified.

Shared Adroit protocol: the same 25 demonstrations/task, native sparse rewards,
256 environments, batch 256 with 128 demonstration + 128 online samples, UTD 1/8
(32 optimizer updates per vector collection), warmup 12,800, 5,000,192 aligned
online transitions and 623,424 updates. No HER, observation/reward normalization,
extra BC loss or offline pretraining. Each native learner keeps its own optimizer,
sampling defaults, with policy/value backbones explicitly matched to two hidden layers of width 256. Native auxiliary timestep embeddings and flow-block counts are retained. There is no tuning on the reported seeds.

- DIPO: official `c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc`, Diffusion policy,
  100 cosine-schedule diffusion steps, action-gradient steps 20, action LR 0.03,
  actor/critic LR 0.0003, gradient clipping 2, action gradient cap 0.1 × action
  dimension, gamma 0.99, tau 0.005, actor target update every optimizer update.
  Native actor accepts batched states for vector collection. Critic replay keeps
  executed actions; a separate mutable action-label buffer persists improved
  policy targets for both demonstration and online states. Indexed assignment
  implements the replacement contract (NumPy `copyto` on advanced-index copies
  would not persist those labels). Neither original dataset files nor critic
  demonstration actions are changed. Native `DiPo.train` and vendor bytes remain
  unchanged. Source: https://github.com/BellmanTimeHut/DIPO
- MEOW: official `b786d27aa9b03e4242ee8904ff884b21fe65e2f7`, original CleanRL
  flow/Q identity and regression update, alpha 0.2, prior log-scales [-5,-0.3],
  LR 0.001, tau 0.005 and gradient clipping 30. Native deterministic flow sampling
  at evaluation. Source: https://github.com/ChienFeng-hub/meow
- MFPO: official `d8b3977d29d4ef2d315e871337e5826f2eb79eb2`, original
  distributional twin-Q and mean-flow update; matched 2×256 policy/value networks with native LayerNorm,
  T=2, initial temperature 0.01, target entropy coefficient -0.5, native entropy
  backup, 16/32 policy/Gaussian proposals and 10-candidate evaluation selection.
  Source: https://github.com/dongxiaoyi-xyz/MFPO

Vendor file hashes and upstream revisions are in `UPSTREAM.json`; existing
licenses/notices are retained. Private adapters are isolated from other
experiments and must not enter any anonymous snapshot.

Evaluation uses the same paired Adroit reset seeds and the native evaluation
rule of each algorithm. DIPO's eval flag disables intermediate diffusion noise
but retains its random initial noise; it is not labeled a deterministic policy.
Record returns and separate any-step, final-step, and 25-success-step metrics,
as well as first online success excluding demonstrations. Native evaluation RNG
is restored. The monitor writes each method's JSON state and CSV curves every
60 seconds. Checkpoints are inference artifacts, not replay-resume promises.

A quick compatibility check performs a real GPU update and native inference for
every method/task shape, verifies exact replay composition, immutable executed
actions and persistent DIPO labels. After the upstream 40-run campaign finishes,
each method additionally receives 4 full-configuration vector smoke checks
(13,568 transitions, 96 updates, 256 environments), before releasing that
method's 20 full jobs. Failure holds later phases for inspection; no automatic
rerun overwrites existing results. A failed upstream run or mismatched completion
marker also holds release. Upstream Supervisor must exit before release.

Supervisor: `adroit-followup-queue-20261110-host1` and three per-method monitor
programs. Queue configuration/state/logs live in `/workspace/heechan-adroit/followup/`.
Code is frozen separately in `/workspace/heechan-adroit/followup-source-20261110`.
All six phase plans are under `/workspace/heechan-adroit/outputs/demo-sparse-followup-*`.

## Superseding architecture profile: matched_256x2

User requested actor and critic 256×2 for all methods. Active iBOLT and SAC already match. DIPO removes the third main hidden layer before training and recreates target networks and optimizers. MEOW widens the prior and four coupling MLPs from 64×2 to 256×2; its two value-shifting MLPs already match. MFPO explicitly overrides both hidden-dimension tuples to (256,256). DIPO and MFPO auxiliary time embeddings retain their native shapes, separately recorded in the audit. Vendor files remain byte-identical. Actual constructed parameters, target networks and GPU updates are checked for all four task shapes.

The old queued native-profile plans are superseded; new plans use the `demo-sparse-followup-256x2-*` prefix and immutable `followup-256x2-source-20261110` source. No old result is relabeled as a matched-architecture result.


## Hammer/Pen verified-data replacement (2026-11-10)

The earlier Hammer/Pen action-replay results are historical previous-dataset results. Current Hammer/Pen runs require the exact verified dataset manifest and independent full saved-episode replay before model construction. Both replacement datasets are expert-policy demonstrations generated from hash-pinned official hand_dapg policies in native Sparse-v2 with the same 200-step limit. They are not human demonstrations. Hammer: first 25 successes from 25 attempts; Pen: first 25 successes from 29 attempts in predefined seeds 100000..104999. All failed attempts, original-state compatibility audits, full trajectories, true termination and timeout flags, initial states, success durations and final-step success are retained. All 25 saved episodes/task replay successfully with zero observation error and identical reward/end/success flags.

All iBOLT/SAC × Hammer/Pen × seed 0..4 runs are new random initializations in separate directories; 5M/256-env/UTD1:8/batch256/demo-online50:50/256x2 and native algorithm settings remain unchanged. The queued DIPO/MEOW/MFPO Hammer/Pen jobs use the same immutable datasets. Final reports require all five seeds and use sample SD (ddof=1), plus complete per-seed learning curves; incomplete cohorts have no final five-seed claim. Evidence and frozen source backups live under ignored outputs/adroit-preparation-20261110/hp-verified25/.
