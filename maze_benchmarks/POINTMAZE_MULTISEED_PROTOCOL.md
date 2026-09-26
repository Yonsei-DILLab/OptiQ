# PointMaze Medium/Hard four-seed extension (2026-09-26)

The user requested four seeds for PointMaze, then excluded Simple and confirmed
that the non-DIPO methods to extend now are OptiQ, SAC, SQL and TD3. MFPO and
MEOW must wait until their best settings are chosen. Preserve all existing
seed-0 runs and their exact frozen source/results; do not rerun seed 0.

DIPO: `POINTMAZE_DIPO_FOUR_SEED_PLAN.json` adds Medium/Hard seeds 1, 2 and 3
(6 runs). Use frozen official DIPO learner source
`4b46c1bfb1f8bb9d93ec0b15e891edf5d9de8c6f`, the same as the running
seed-0 controls. Preserve native BellmanTimeHut/DIPO actor, 100 diffusion steps,
20 action-gradient steps and its optimizer. Each new run has 2048 vector
Environments, batch 4096, 8192 warmup transitions, 128 updates per 2048
collected transitions, 1,001,472 transitions and 62,080 learner updates.
Preflight: 10,240 transitions and exactly 128 updates. Final direct-policy
PointMaze evaluation: 500 episodes, with original sparse +100 task/goal map.

Other baselines: `POINTMAZE_BASELINE_FOUR_SEED_PLAN.json` adds Medium/Hard
seeds 1, 2 and 3 for OptiQ, SAC, SQL and TD3 (24 runs). Frozen learner source
`bfb06944705685a7dea84558aba40500c22349b9` retains the same task
geometry/reward and native algorithm implementations as the 1M seed-0
controls. Use 256 vector environments, batch 4096, 8192 warmup transitions,
16 updates per 256 collected transitions, 1,000,192 transitions and 62,000
learner updates. Preflight: 8448 transitions and 16 updates. All methods have
200k evaluations and a final 500-episode direct-policy evaluation. OptiQ
uses fresh random-z mu-only sampling with conditional sigma removed; all
other methods use their native sampling. OptiQ Medium T=3 follows the
1M seed-0 control (success 500/500, removal SR5=0.985). OptiQ Hard T=10
follows the complete T sweep: success 499/500, removal SR5=0.958 versus
T=3 SR5=0.854; it is the best observed Hard diversity setting. Other
hyperparameters retain native defaults from frozen source.

The generic `deadline_queue` now passes `job.seed` to the learner and audits
it in the saved config, defaulting to zero for historical plans. This
controller change is post-launch for old experiments and does not alter any
frozen learner. Commit controller/plan/protocol before each new preflight or
main run. Each host campaign records controller and learner commits separately.
Use independent guarded GPU workers and locks; do not interrupt other running
learners. Start the six DIPO jobs first and verify their preflights/claims,
then start baseline workers on eligible slots so they immediately backfill
when a slot frees. Failure pauses that host's queue, preserves live jobs and
all logs, and requires investigation before any restart. Do not launch
MFPO/MEOW multi-seed jobs or additional Simple/Way seeds in this campaign.

At completion, collect config, raw goal IDs and trajectories, intermediate
and final metrics, checkpoint/replay, source/proof manifests and SHA-256 to
local artifacts. Report each policy's four seeds separately and aggregate
success, reachable goals, removal SR5 and obstacle SR5 as mean and sample SD.
Do not treat multiple seeds' trajectories as if one policy were multimodal.
