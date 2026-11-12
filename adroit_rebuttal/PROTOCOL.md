# Adroit: demonstrations plus online RL, sparse rewards

Work and execution remain on non-anonymous `heechan-rebuttal` sources. The main
repository's iBOLT loss, proposal, density correction, TD target, and DACER code
are used without changes in the F1/A1 campaign. Code is maintained on the
non-anonymous `heechan-rebuttal` branch; anonymous export remains prohibited.

The review asks for complex sparse-reward Fetch or Adroit validation. This
experiment addresses sparse Adroit **with demonstrations**; it does not establish
unaided sparse-reward exploration. Demonstrations are available equally to both
methods, and all five training seeds are reported, including failures.

## Current expert-policy comparison (2026-11-12)

Each of Door, Hammer, Pen, and Relocate uses one fixed dataset of 25 distinct
expert-policy episodes. Both learners and all training seeds for a task use the
same dataset hash. Every episode succeeds under the native 200-step Sparse-v2
limit. The loader replays the saved initial states and actions and verifies
observations, rewards, termination, timeouts, and success before model creation.
The earlier human-action replay results use their original dataset and protocol.

New SAC runs use the shared `baselines/jax_sac.py` learner, with paired reset and
policy seed schedules. The numerical check is
`rebuttal_tools/check_fetch_jax_sac.py --adroit-task <task>`.

The current **A1** protocol supersedes the historical human-action-replay table
below: actor and critic 256x2, critic LayerNorm enabled, vec256/four workers,
batch256 with 128 expert and 128 online transitions, UTD1/4 (64 updates after
each vector step), gamma0.99, tau0.005, and learning rates3e-4. Use random learner
initialization, no BC pretraining, no HER or observation/reward normalization,
12,800 warmup transitions, and 5,000,192 total transitions (1,246,848 updates).
The environment and demonstrations both use the native 200-step limit.

Current primary evaluation uses fresh stochastic latent z and conditional mean
for iBOLT, and the deterministic actor for SAC. Task success uses the existing
native-success-based evaluator, with 10 episodes per regular evaluation and
100 at the final step. Preserve paired environment and policy seed schedules.
Do not mix the historical zero-z primary mode with this comparison.

Temperature pilots use seeds5/6, followed by independent seeds0-4 confirmation.
DACER-target and N/M comparisons are separate experiments. The fixed-alpha=0
JAX SAC comparison uses seeds0/1 only and does not establish a five-seed result.
The source snapshots and recorded protocols are indexed in
`rebuttal_tools/campaign_snapshots/20261112/README.md`.

## Historical human-action-replay protocol

The following paper adaptations, table, evaluation mode, and server allocation
describe the original experiment. They are retained for provenance and do not
override the current A1 settings above.

## Published sources and adaptations

- Rajeswaran et al., **RSS 2018**, *Learning Complex Dexterous Manipulation with
  Deep Reinforcement Learning and Demonstrations*. Supplies the four Adroit tasks
  and 25 human demonstrations per task: https://www.roboticsproceedings.org/rss14/p49.html
  Paper: https://www.roboticsproceedings.org/rss14/p49.pdf
  Official demonstrations: https://github.com/aravindr93/hand_dapg/tree/9a1de23df6574e1b94e1da742e3326eb1f61c013/dapg/demonstrations
- Ball et al., **ICML 2023**, *Efficient Online Reinforcement Learning with Offline
  Data*, section 4.1: sample half of every minibatch from offline data and half
  from online replay, without offline pretraining or an imitation loss:
  https://proceedings.mlr.press/v202/ball23a.html
  https://proceedings.mlr.press/v202/ball23a/ball23a.pdf

We adopt the RLPD **data-sampling protocol**, not its full algorithm. In
particular, we keep each learner's original networks and losses; do not add
RLPD's 10-critic ensemble, LayerNorm or change SAC's entropy backup. RLPD's main
sparse Adroit experiments cover three tasks with human and behavior-cloned data;
our four-task human-only comparison is not an exact reproduction of those curves.
The requested 256 environments and UTD 1/8 are our execution choices, not paper
hyperparameters. Temperature is fixed at 0.25 before seeing these results; there
is no tuning on reported seeds. SAC uses its standard automatic temperature.

## Frozen protocol

| Setting | Value |
| --- | --- |
| Tasks | Door, Hammer, Pen, Relocate, `AdroitHand*Sparse-v2` |
| Reward | Native sparse reward, no dense shaping or reward normalization |
| Methods | Existing iBOLT + demonstrations; SB3 SAC + demonstrations |
| Human trajectories | 25 per task, same data for both methods and every seed |
| Seeds | 0–4 |
| Online target | 5,000,000 transitions; actual aligned budget 5,000,192 |
| Parallel environments | 256, grouped across 4 CPU workers |
| UTD | 1/8; collect 256 transitions then perform 32 optimizer updates |
| Batch | 256 = 128 demonstration + 128 online transitions at every update |
| Warmup | 12,800 online transitions; no offline pretraining or BC warm-start |
| Full-run updates | (5,000,192 − 12,800) / 8 = 623,424 |
| Online replay capacity | Approximately 1,000,000, aligned to 256 environments |
| Horizon | Native online TimeLimit 200; timeout transitions bootstrap |
| Actor / critic | Existing two-layer 256-unit networks; plain twin Q |
| LR / gamma / tau | 0.0003 / 0.99 / 0.005 |
| iBOLT | T=0.25, N=M=64, log-sigma [-5,-1], initial -1; existing DACER |
| SAC | Native Gaussian actor, automatic entropy coefficient/target entropy |
| HER / observation normalization | Disabled |
| Evaluation | Initial, every 50,176 transitions, and after the final update |
| Evaluation episodes | 10 per intermediate checkpoint, 100 at completion |

The official DAPG pickle hashes are pinned before deserialization. Their action
sequences are executed from their supplied initial simulator states in the
current sparse v2 environment, generating consistent `(s,a,r,s',done)` tuples.
We preserve all 25 trajectories without selecting the best demonstrations. Original
demonstrations can extend beyond the 200-step online horizon; the full sequences
are replayed in the underlying MDP without artificial resets. Only true MDP
termination masks are stored; demonstration endpoints are not treated as failure
terminations. Dataset manifests record trajectory lengths, transfer success and
initial-observation differences, and block release if a task has no successful
transferred demonstrations. Report this conversion explicitly; these are not
unchanged original D4RL transition files.

Gymnasium-Robotics is frozen at upstream
`b8da02a89191daf8f004e7f7510ff655f7a90100`, which registers v2. Its package version
still reports 1.4.2; the PyPI 1.4.2 code differs. Every file in the installed wheel
is checked against the wheel built from the pinned source archive. The server's
verified native assets may be reused when reconstructing that exact package file
set. Runtime: Python 3.11, Gymnasium 1.2.2, MuJoCo 2.3.7, SB3 2.7.0, JAX 0.6.2,
Flax 0.10.6, Torch 2.8.0+cu128, NumPy 1.26.4. Full dependency freeze is stored on
the server and backed up locally. The private venv uses a filesystem copy of the
existing compatible base and the pinned environment package; it does not modify
running Fetch environments.

## Evaluation and monitoring

Record returns, native `info['success']` at any step, final-step success, and at
least 25 successful steps (not necessarily consecutive). Keep these definitions
separate. iBOLT reports zero-latent mean, stochastic-latent mean, and full mixture
sampling. SAC reports deterministic and stochastic sampling. The native reference
rule is zero-latent mean for iBOLT and deterministic actor for SAC; full-policy
results remain separately available to address the review's evaluation question.
All methods and modes receive paired reset seeds. Evaluation restores policy keys
and Python, NumPy and Torch RNG state. Online first-success discovery counts
exclude all demonstrations and refer to the end of a 256-transition collection
batch. Monitor metrics/steps and errors separately from achieved policy success.

Eight GPU smoke checks use **256 environments**, warmup 12,800, 13,568 total
transitions and **96 real updates** each. They verify minibatch composition,
finite parameters, evaluation RNG preservation and checkpoint inference.
Environment checks cover native task dimensions, seeded trajectories, episode
boundaries and EGL rendering. Full execution is released only after all checks
pass. Training failures are preserved and are not automatically retried.

Server: `ibolt-server-1`, root `/workspace/heechan-adroit`, Python `venv/bin/python`.
Supervisor runs setup/validation, the campaign and a persistent monitor. At most
one Adroit learner runs per GPU. Existing tasks remain active; admission permits
sharing only below 8,192 MiB memory use and 60% sampled GPU utilization. The
128-logical-CPU server provides headroom for grouped CPU simulation. These admission
thresholds do not guarantee constant training speed; monitor actual progress.


## Hammer/Pen verified-data replacement (2026-11-10)

The earlier Hammer/Pen action-replay results are historical previous-dataset results. Current Hammer/Pen runs require the exact verified dataset manifest and independent full saved-episode replay before model construction. Both replacement datasets are expert-policy demonstrations generated from hash-pinned official hand_dapg policies in native Sparse-v2 with the same 200-step limit. They are not human demonstrations. Hammer: first 25 successes from 25 attempts; Pen: first 25 successes from 29 attempts in predefined seeds 100000..104999. All failed attempts, original-state compatibility audits, full trajectories, true termination and timeout flags, initial states, success durations and final-step success are retained. All 25 saved episodes/task replay successfully with zero observation error and identical reward/end/success flags.

All iBOLT/SAC × Hammer/Pen × seed 0..4 runs are new random initializations in separate directories; 5M/256-env/UTD1:8/batch256/demo-online50:50/256x2 and native algorithm settings remain unchanged. The queued DIPO/MEOW/MFPO Hammer/Pen jobs use the same immutable datasets. Final reports require all five seeds and use sample SD (ddof=1), plus complete per-seed learning curves; incomplete cohorts have no final five-seed claim. Evidence and frozen source backups live under ignored outputs/adroit-preparation-20261110/hp-verified25/.
