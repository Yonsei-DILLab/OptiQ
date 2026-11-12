> **Cancelled by the user on 2026-11-10.** Search and confirmation were stopped;
> automatic-alpha reference runs remain active. The extra baseline queue gate
> was removed. Configuration and partial results below are retained as history.

# SAC fixed-alpha grid (Adroit sparse, demonstrations + online RL)

The existing automatic-alpha iBOLT/SAC campaign and its immutable source remain
unchanged. This separate SAC-only grid was requested after poor SAC results and
very large learned alpha/critic loss were observed in Door. Those observations
motivate the experiment; they do not prove alpha is the sole cause.

SB3 `ent_coef` is the entropy coefficient alpha. A numeric value fixes alpha and
creates no entropy-coefficient optimizer; `auto` learns alpha. The target entropy
is only active under automatic tuning. Official API:
https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/sac.html#parameters

- Fixed grid: **0.01, 0.03, 0.1, 0.3, 1.0**.
- Four tasks: Door, Hammer, Pen, Relocate, native sparse v2 rewards.
- Search seeds **10 and 11**, 5 alpha values × 4 tasks × 2 seeds = **40 runs**.
- After selecting one alpha per task, confirmation seeds **0–4**, **20 runs**.
- Each full run: 5,000,192 aligned online transitions (target 5M), 256 envs,
  warmup 12,800, UTD 1/8 (32 updates/collection), 623,424 updates, batch 256 with
  128 fixed demonstration and 128 online transitions.
- Same data, models, learning rates, rewards, evaluation seeds and native SAC
  losses as the existing auto reference. No HER, normalization or BC pretraining.

Selection is fixed before inspecting these runs: highest mean **final
100-episode deterministic success rate across seeds 10 and 11**; ties use
0–5M success AUC, final mean return, then lower alpha. Both search seeds must
complete for a candidate to be eligible. Failed candidates are retained and
reported, rather than averaging only their successful seeds. Confirmation
failures do not trigger reselection on reporting seeds. Search seeds are never
pooled with the final five-seed result. Stochastic-policy evaluations are also
retained but do not replace the prespecified primary selection criterion.

No automatic early stopping or best-checkpoint reporting is used. The fixed
five-value grid is an additional SAC tuning budget; comparisons should disclose
this budget rather than claim equal hyperparameter search across methods.
The original auto results remain a separate reference.

Five vector GPU smoke runs (Door, seed 831, one per alpha) each collect 13,568
transitions and execute 96 optimizer updates at the full batch and vector sizes.
They assert that the entropy optimizer is absent, alpha stays fixed throughout
updates, evaluation preserves training RNG and checkpoint inference restores.
All five alpha variants must have identical initial actor parameter hashes.
Existing vector/environment checks cover all four Adroit shapes.

Maximum **two additional grid learners** run concurrently, one grid learner per
GPU, sharing with the active original experiment only while GPU memory is below
8 GiB and sampled utilization is at most 60%. Grid learners use their own GPU
locks and all run through Supervisor. Other experiments remain running.

Execution: smoke → 40 search runs → alpha selection → 20 confirmation runs.
The existing DIPO → MEOW → MFPO continuation additionally waits for successful
grid confirmation, then retains its original 40-run iBOLT/SAC prerequisite.
Failures hold progression for review, with no automatic replay resume or output
overwrite. Search and confirmation have separate 60-second monitor JSON/CSV logs.

Server root: `/workspace/heechan-adroit/sac-alpha-grid/` (config, state, selection,
controller log). Immutable source: `/workspace/heechan-adroit/sac-grid-source-20261110`.
Plans/results: `/workspace/heechan-adroit/outputs/sac-alpha-grid-{smoke,search,confirmation}-20261110/`.
Supervisor: `adroit-sac-alpha-grid-20261110-host1`, plus search/confirmation monitors.
