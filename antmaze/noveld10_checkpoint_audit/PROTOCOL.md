# NovelD10 checkpoint audit before any new training

2026-09-22 user: Hold new training. Inspect saved policies to decide whether coefficient10 is appropriate.
No learner updates, no checkpoint overwrites. Cancelled training sources/logs remain immutable.
Evaluate one training seed0 per policy,100 fixed-full-state episodes for direct-policy and native modes;
matched checkpoint steps and common environment/policy RNG seeds between coefficients.
No external DACER action noise or NovelD reward in evaluation. OptiQ native=random-z mu-only;
policy=random-z with conditional sigma. SAC native=mean, policy=sample. MFPO native=Q-best-of10,
policy=direct sample. Do not mix policies/seeds to claim within-policy diversity.

v1 OptiQ300k/SAC200k/MFPO300k, v2 OptiQ100k/SAC100k: compare c10 against c.01 at the same steps.
v3 OptiQ100k c10 versus c.01: c10 comes from the earlier completed coefficient sweep.
v4 has no c10 checkpoint and cannot support a conclusion about coefficient10.

Verify checkpoint files/SHA256/full-state/model restoration/replay dense rewards. Assert inference
leaves model and learner RNG unchanged. Retain raw xy/actions/observations and training coverage.
Report training coverage and goal/corridor visitation separately from final policy trajectories;
success/return are secondary. Do not claim coefficient10 optimal or extrapolate to all four mazes.
Commit diagnostic source before running. New DIPO/reduced evaluation training remains on hold.
