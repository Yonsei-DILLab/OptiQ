# PointMaze Medium baseline sensitivity grid

This is a **separate**, preregistered seed-0 check of the longer PointMaze
comparison. It does not mutate, stop, resume, or replace the frozen 21-policy
1M/2M/3M campaign (source `d710cc668a61e69b52f0b71bbe724d6a85c499fb`).
Only the original DrAC Medium map is used here: fixed center reset, four goals,
upstream MuJoCo physics, sparse +100 on reaching any goal, 300-step horizon.
The source and exact ten jobs are fixed in `POINTMAZE_MEDIUM_BASELINE_GRID.json`
before launch. All ten start fresh at seed 0 and collect 2,000,128 transitions.

The current long campaign uses 256 parallel environments, batch 4096 and 16
optimizer updates per 256 collected transitions, or optimizer UTD=0.0625. The
published DrAC implementation sets `train_ratio=1` and batch 256. Batching
4096 with 16 updates preserves 256 replay samples per new transition, but
reduces the number of parameter updates by 16. At Medium 2M, the current
profile performs 124,496 updates. This grid doubles it to 32 updates per
collection (UTD=0.125, 248,992 updates) while keeping batch 4096. The longer
run at UTD16 is the control. UTD32 approximately matches the paper's *200k
update count* for Medium, but processes many more replay samples and is **not**
an exact paper reproduction. All other shared settings, including replay 1M,
warmup 8192, gamma .99, eval/reset rules and final 2,000-episode policy and
obstacle rollout, remain fixed.

Seven UTD32 jobs retain each method's original baseline adapter settings.
Three additional one-factor settings probe possible task mismatch:

| Method | Existing setting | Additional setting | Rationale |
| --- | --- | --- | --- |
| SQL | 16 actor/value particles | 32 each | The DrAC PointMaze SQL launcher uses 32 SVGD particles; this JAX port remains a different implementation. |
| MEOW | fixed alpha 0.2 | fixed alpha 1.0 | Check whether its original MuJoCo entropy weight is too weak against a +100 success signal. |
| MFPO | target entropy -0.5 per action dimension | 0.0 per dimension | Check whether a higher entropy target prevents policy-mode loss. |

DIPO keeps the original DDiffPG five-step diffusion actor, mixed exploration
noise, action-gradient iterations, distributional support [0,120] and numerical
projection guard. Its one-goal policy in the short run is **not** evidence that
the diffusion sampler was accidentally deterministic: the original sampler
starts from fresh Gaussian noise even during evaluation, but DIPO has no
explicit entropy or diversity objective. TD3 is deterministic by design. SAC
uses learned entropy, and its target is -2 for this two-dimensional action.
Artificially adding evaluation-only action noise would not establish learned
policy multimodality, so this grid does not do so.

Every job must pass a real 8,448-transition/32-update preflight on its assigned
GPU, verifying finite raw policy/obstacle rollouts and config, before its fresh
main run. GPU locks and hardware-idle guards prevent interference with the
existing campaign. The user's reserved `vast-heechan-180` GPU2 is excluded;
GPU1 on that host is also excluded after a CUDA launch failure left its device
handle unreadable on 2026-09-26. Only six healthy RTX5090 slots are eligible,
and each queue worker may
backfill as soon as its slot is free; no all-campaign completion barrier.
The four RTX4090 GPUs are not used. A failed setting pauses pending jobs on
that host and retains its partial source/log/result. No automatic retry or
hyperparameter substitution occurs.

Final comparison must report direct sampled-policy success, per-goal counts,
number of reached goals and raw trajectories for each named variant beside
its UTD16 control. Include 2,000-episode uncertainty for each goal frequency,
but acknowledge training seed0 alone cannot measure training variability.
Do not silently select a variant by the test result or relabel the tuned
profile as the original baseline. The purpose is to determine whether the
one-path result is sensitive to update count or these three method-specific
settings. Wider grids or full Simple/Hard tuned runs require subsequent review.

Primary provenance: [DrAC configuration](https://github.com/PneuC/DrAC/blob/main/rl/config.yaml),
[DrAC SQL launcher](https://github.com/PneuC/DrAC/blob/main/train_svgd.py),
the pinned `antmaze/ddiffpg/cfg/algo/dipo_algo.yaml`,
`gmm40-baseline/meow/cleanrl/cleanrl/meow_continuous_action.py`, and
`gmm40-baseline/MFPO/configs/mfpo_config.py` in this repository.
