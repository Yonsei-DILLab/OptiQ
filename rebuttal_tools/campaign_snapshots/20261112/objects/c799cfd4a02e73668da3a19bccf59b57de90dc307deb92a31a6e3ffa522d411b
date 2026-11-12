# Fetch PickAndPlace rebuttal campaign

User scope: FetchPickAndPlace-v4 only. Other Fetch tasks are not launched.
Dedicated non-anonymous package and Python overlay; original running experiments
and their environments are unchanged. See AGENTS.md for the private-source boundary.

Initial runs: SAC, IDAC-SIA, IDAC-noDGN, MFPO, MEOW × seeds 0–4 (25), plus
iBOLT temperatures 0.05, 0.1, 0.25, 0.5, 1, 5, 10 × seeds 0,1 (14).
All 39 are submitted together across both 4-GPU hosts, admitted as memory allows.
The controller bounds concurrent compiler starts and keeps 3 GiB GPU headroom;
it does not wait for the existing experiments to finish.

Protocol: one environment, one million transitions including 5K warmup, UTD=1,
batch=256, replay=1M, original sparse rewards (-1/0), original 50-step horizon.
Input is observation concatenated with desired_goal (28D); action is original 4D.
No HER, reward shaping, normalization, demonstrations, or environment changes.
Time-limit truncations bootstrap from the actual final state. Only true
terminations stop bootstrapping. Train/evaluation RNG streams are separated.

iBOLT imports the existing MuJoCo learner and exploration regulator unchanged:
root configs/train.yaml, N=M=64, actor/critic 256×2 GELU, lr=3e-4,
gamma=.99, tau=.005, scale [-5,-1], DACER behavior exploration multiplier=.1.
Only temperature is swept. Paired MuJoCo evaluations retain zero-z centers and
resampled-z centers, without conditional Gaussian or external exploration noise.
Resampled-z center success is the primary temperature-selection metric.

Baselines use their frozen native profiles and architecture; no tuning:

| Method | Native configuration |
|---|---|
| SAC | SB3, 256×2 ReLU, lr3e-4, automatic entropy, deterministic evaluation |
| IDAC-SIA | Official zhougroup/IDAC, semi-implicit actor, 256×2, critic noise5, 32 quantiles, actor noise5/10 auxiliary samples, lr3e-4, automatic entropy |
| IDAC-noDGN | Same SIA/entropy/optimizer/critic width+LN; critic noise dimension0, one scalar output, plain twin-Q MSE; explicit local ablation, not upstream CLI |
| MFPO | Pinned author configuration: T2, proposals16/32, divergence2, 256×3 GELU/LN, automatic entropy; official evaluation candidate selection10 |
| MEOW | Pinned CleanRL flow: 2 flow blocks, hidden64, scale hidden256, alpha.2, lr1e-3, prior[-5,-.3], gradient clip30; native deterministic evaluation |

These are native/common MuJoCo profiles, not a claim that every baseline's
unmodified upstream CLI has identical hyperparameters. IDAC uses the requested semi-implicit variant; the common collector uses5K
warmup (the upstream IDAC CLI uses10K). SB3 uses the Blackwell-compatible pinned version. Vendor code is
byte-identical; the adapters only generalize dimensions and connect the task.
SOURCE.json and per-run resolved configurations record the exact provenance.

Evaluation every5K: 10 fixed-seed episodes, plus100 episodes at the end.
Primary success: info.is_success at the final step. Anytime success, episodic
returns, lengths, per-episode results, checkpoints every100K and final replay
are also saved. Both iBOLT mean-action modes are kept separate.

Selection: when both seeds of all seven temperatures finish, compare mean success
across the last ten pre-final checkpoints (950K–995K) and two seeds. Tie-break by
success learning-curve mean, then closest to temperature1. Require nonzero late
success; if every temperature has zero late success, record needs-review and
do not automatically call a setting good. The selected temperature keeps seeds
0/1 and adds2/3/4, yielding five independent training seeds with disclosed overlap
between tuning and reporting seeds. Three additional runs bring the total to42.
This is not an independent held-out hyperparameter-selection evaluation.

Both host controllers persist plan.json, ready.json, metrics.json and logs under
outputs/fetch-rebuttal/. Supervisor keeps the controllers independent of SSH.
Each host validates all six real optimizers on its GPU at the actual batch256
before it dispatches main training. Smoke runs are excluded from reported runs.

## Authorized followup queue

For each of SAC, IDAC-SIA, IDAC-noDGN, MFPO, MEOW, wait until all five seeds
of PickAndPlace finish successfully. Then run the same method on Push (five
seeds), then Slide (five), then Reach (five), using a five-seed barrier at each
step. Methods advance independently. Each newly released method/task passes a
real GPU optimizer preflight, including Reach's different observation dimension.
Failures do not silently release later tasks. Other-task iBOLT sweeps are not
included in this request. This adds75 deferred main runs; initial39 + deferred75
+ selected-iBOLT3 =117 maximum main runs. GPU smoke runs are excluded.

Host1 owns the full SAC seed groups and all iBOLT temperature runs. Host2 owns
all seed groups for IDAC-SIA, IDAC-noDGN, MFPO and MEOW. Thus no seed barrier
requires cross-host credentials or keeping the local laptop awake.
