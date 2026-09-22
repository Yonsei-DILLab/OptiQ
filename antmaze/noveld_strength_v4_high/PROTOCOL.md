# AntMaze v4: NovelD 50 and 100, exploration-first

Latest user authorization (2026-09-22): use the remaining two GPUs for v4
NovelD coefficients50/100 and later compare exploration range against0.01.

Exactly two fresh OptiQ v4 seed0 runs of100k environment interactions/95k
learner+RNDupdates, server180 GPU2/3. Keep the concurrent v3 runs on GPU0/1
and all other campaigns/sources/results intact. This protocol only overrides
the task and coefficient. All training code/settings match the approved v3
50/100 protocol, source36aab085bcfe8219997e1bc21e3ad8fe55f266b0. Default NovelD
coefficient remains .01. No extra seeds/environments/steps or automatic restart.

Commit before execution, then each setting272step smoke→new-process280step
resume, coefficient linearity/default compatibility checks, initialmodel/RND
and non-ablationconfig checks before production. Compare to the existing v4
seed0 .01 campaign from source19fc37a; its reference metadata is copied from
server199 to /home/heechan/OptiQ-ops/references/antmaze-v4-noveld001-19fc37a.
Exclude only coefficient, source/output paths and budget/profile metadata from
config comparison. Do not rerun the old campaign's preflights or training.

The .01 baseline was scheduled for1M. For exploration comparison use its
100k checkpoint/replay, not1M coverage versus100k coverage. Keep original
checkpoint manifest and verify its hashes; reconstruct visited bins from
the first100k chronological transition positions at0.5m resolution. Evaluation
rollouts are not part of training coverage. Trainingbudget is the comparison
budget, independent of the longer baseline's eventual planned endpoint.

Primary outcomes: cumulative visited bins/area at equal100k, occupancy heatmaps,
exploration depth along both corridors/goal directions, and concentration of
visits. Success/return/firstgoal are secondary, not the selection criterion.
Report trainingcoverage separately from finalpolicy rollout distributions.
Do not promise that more steps necessarily produce success; one seed is a probe.

W&B OptiQ/gmm-trg, group antmaze-v4-optiq-noveld-high-100k-s0-20260922.
Evaluation unchanged: every25k10episodes/mode and final100episodes/mode/reset;
native randomz mu-only,policy randomz+conditional sigma,zero_z separately.
No externalDACERnoise or intrinsicreward in evaluation. Natural/fixed v4 start
states are the same; do not pool them as200independentstarts.

Use supervisor and GPUlocks. Failures holdpending, preservelive, noautorestart.
Save allreplay/model/optimizer/NovelD/DACER/RNG/simulator state andrawrollouts.
Collect using antmaze.noveld_strength_v4_high.collect_report --source <commit>;
archive locally at artifacts/antmaze_noveld_strength_v4_high_100k andverify.
Create exploration-first comparisons by postprocessing without changing the
immutable learning sources or original summaries.
