# DACER-off AntMaze comparison — 2026-09-24

User-requested additions, not replacements. Preserve the running 27-job sweep.
v1/v3/v4, seeds 0/1/2: dense negative nearest-goal distance + NovelD OFF at
T=3,5 (18); official sparse reward + NovelD coefficient .01 at T=1,3,5 (27).
All 45 disable DACER only; do not remove the stochastic policy's own noise.
Keep frozen prior profile: actor/critic 256x3, LR3e-4/5e-4, N=M64, beta1,
log sigma[-5,-1] initial-1, 256 envs, B4096, 8 updates/256 transitions,
8192 warmup, native 3M/4M/5M post-warmup budgets, random-start evaluations
every250k (40 episodes), final100 and intermediate policy checkpoints.
W&B OptiQ/antmaze, group antmaze-dacer-off-dense-sparse-20260924.
Separate committed snapshot and controller; no changes to old frozen code.
Prior pending jobs take priority. Reuse validated runtime via symlinks.
Every job runs its own real-GPU preflight, with explicit config checks for
DACER=false, reward, NovelD coefficient, updates, seed and model architecture.
Failure holds pending jobs; never silently restart training.
