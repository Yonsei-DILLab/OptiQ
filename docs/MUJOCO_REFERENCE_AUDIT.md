# Alignment with the supplied actual run (2026-09-07)

The user confirmed that the pasted actual execution record takes precedence
over the older generic `critic-dime` experiment design. Its configuration is
also represented by `scripts/run_dog_landscape_rerun.sh` in `critic-dime`.
The older two-seed design and the unrelated `dev` MuJoCo budgets are not the
reference for this sweep.

The user subsequently requested a four-seed sweep. All remaining parameters
continue to use the supplied actual run as the reference.

## Existing seed evidence and requested expansion

All remote branch tips were fetched and their launch/configuration files inspected.
The supplied record itself documents seed 1; launch settings show configured seeds,
not proof that every configured run completed.

| Source | Recorded / configured seeds |
| --- | --- |
| Supplied actual run | **1** |
| `critic-dime:scripts/run_dog_landscape_rerun.sh` | 1 |
| `critic-dime:EXPERIMENT_DESIGN.md` | 1,2 |
| `heejoon:scripts/run_four_way_proposal_recovery.sbatch` and temperature sweep | 1,2,3 (`array_index % 3 + 1`) |
| `dev:launch/mujoco_5env_5seed.sh` and SAC/TD3/SQL launcher | 1,2,3,4,5 (`array_index % 5 + 1`) |

No dedicated four-seed suite was found in the inspected remote-branch launch
settings. The requested default is **1,2,3,4**, preserving the actual reference
seed and following the existing 1-based numbering. Seed 4 is an added replicate
relative to the DIME-related suites. This does not import `dev`'s algorithm or
training budgets. Direct entry remains seed 1; the shell sweep explicitly passes
each of the four seeds. All 20 environment/sampler/beta combinations receive the
same four seeds, giving **80 runs** (80M training environment steps).

## Corrections to the initial heechan preparation

| Setting | Initial heechan (`2968f8f`) | Supplied record / corrected default |
| --- | --- | --- |
| Sweep seeds | 1,2,3,4,5 | **1,2,3,4**, explicitly requested expansion of reference seed 1 |
| Training budget | Ant 3M, Humanoid 5M | **1M for both** |
| Direct-entry sampling default | exact | **stratified**; sweep explicitly selects both |
| Evaluation environment seeds | New fold-in sequence with offset 10000 | Original `jax.random.randint(jax.random.key(seed), (10000000,), 0, 2**30-1)` seed list |
| Evaluation action RNG | Independent stream, training RNG restored | Original policy RNG consumption; no isolation/restoration |
| Evaluation timing | Step 1, intervals, extra final evaluation | Original callback: step 1 and every 5k; no extra off-schedule final evaluation |
| Training logging | Extra flush every 1000 environment steps | Original `DIME.learn(log_interval=1)`, per completed episode |
| Progress bar | Off | On, as in the original runner |

The original evaluation callback is inherited directly. The only addition to
that callback is writing the already collected return/length arrays to disk.
Its environment seeds, stochastic policy calls and evaluation timing are not
reimplemented. Since evaluation advances the policy RNG in the original code,
this also preserves its effect on subsequent training actions.

## Parameters already matching the supplied run

- Actor: three 256-wide GELU layers; critic: two 2048x2048 ReLU categorical
  networks with 101 atoms; critic batch renormalization, momentum 0.99,
  warmup 100000, no layer normalization or dropout.
- Replay 1M; batch 256; random warmup 5000; UTD 2; policy delay 1;
  discount 0.99; tau/policy_tau 1.0; no model resets.
- Actor/critic learning rate 0.0003; Adam betas (0.9,0.999)/(0.5,0.999).
- Maximum-entropy alpha 0; critic distribution-entropy coefficient 0.005.
- N=16, R=5; anchors enabled (16 anchors and 64 random proposals).
- Proposal std 0.1, clip 0.15; Q temperature 0.25; source Q is the twin mean;
  uniform-action reference and proposal-density correction enabled.
- Sinkhorn epsilon 0.05, 30 iterations; argmax assignment and squared-error
  actor distillation; TD noise std 0.2 and clip 0.5.
- JIT enabled; stochastic evaluation, 10 episodes at step 1 and each 5000;
  checkpoints every 50000 steps, with the existing first-trained-state save.
- The same DIME/SB3 collection and replay path is used. Environment-required
  observation/action dimensions, action rescaling and replay space type differ
  from Dog. Actor/critic model implementation files remain unchanged.

## Intended differences

- Ant-v4 / Humanoid-v4 instead of DMC Dog, with the user-approved critic support
  [-1600,1600] instead of [-200,200].
- Experiment axes: stratified vs IID mixture allocation and fixed density beta
  0.1,0.25,0.5,0.75,1.0, with the requested four seeds 1,2,3,4.
  Adaptive beta stays off. Default total: **80 runs**.
- Per-environment output/W&B names, correct environment/device metadata, explicit
  online authentication, result persistence and final checkpoint artifact upload.
- Diagnostics group exact-mixture samples by their actual generating component;
  this does not change the actor objective, update or returned RNG key.

## Regression evidence

`tests/data/supplied_dog_reference.json` contains only the hyperparameter portion
of the supplied run (no credentials, telemetry, or private output paths).
Two tests recursively compare every recorded parameter against the resolved
Ant/Humanoid configs; the only excluded recorded hyperparameters are the approved
critic support endpoints. Tests also check direct-entry defaults and all 80 sweep
configurations, including all four seeds in every environment/sampler/beta group.

Two actor regression tests load the original `critic-dime` implementation at
`8b8fee13b2cbc4d90183d7b803fce033eadf0a19` and compare both samplers' returned RNG
keys, actor loss, parameters and optimizer state on identical inputs. Keys match
exactly, and numeric updates match at rtol 1e-6 / atol 1e-7.

The evaluation regression compares the inherited callback against the original:
the complete seed arrays, scheduled timesteps, episode returns, and resulting
policy/noise keys match exactly on the same test environment and initial state.
At test intervals of 4 steps, a 6-step run evaluates at [1,4], not at step 6.
All six targeted reference/configuration regression cases passed.
After the four-seed expansion, the task-table regression passed again across all
80 resolved configs. The actual shell `--list` produced 80 commands (20 per seed),
and the four-worker partition covered each command once. `--seeds 1` still
produced the 20 reference-seed commands. Shell syntax and `git diff --check` passed.

The changed evaluation path was also tested with full-size networks, batch 256,
12 training steps, warmup 8, one evaluation episode every 4 steps, and checkpoint
interval 10. These are explicitly tagged smoke runs, not benchmark results:

- [Humanoid-v4 stratified, beta=0.1](https://wandb.ai/sae_project/optiq_dime_mujoco_v4/runs/d11dofys)
- [Ant-v4 exact, beta=1.0](https://wandb.ai/sae_project/optiq_dime_mujoco_v4/runs/mq8mx9op)

The W&B API confirmed both runs finished with 12 environment steps, 8 updates,
and the last evaluation at step 12. Their saved evaluation timesteps are
[1,4,8,12]. These smoke runs used reference seed 1 before the four-seed expansion;
the expansion was validated through the configuration table and launcher checks.

Matching the protocol does not imply identical trajectories across different
environments, hardware or library builds. The JAX/DIME package versions and GPU
backend are recorded in every W&B run. No full benchmark was started.
