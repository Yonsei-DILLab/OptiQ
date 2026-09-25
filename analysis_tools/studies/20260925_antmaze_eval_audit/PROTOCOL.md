# Historical AntMaze evaluation reset audit

Read-only reevaluation of the completed v3/v4 OptiQ critic-dynamics controls
at 508,416 transitions. Load the SHA-verified final checkpoint with its frozen
source. Keep the trained actor, action sampling, reward and maze unchanged.

Compare original identical full-state evaluation (`fixed-full`) with the
upstream training reset distribution (`native-pose`): xy remains exactly(0,0)
but initial pose/velocity vary. Optionally use `random-xy` only as a separate
out-of-distribution sensitivity check; v3/v4 did not train with random xy.

Evaluate the direct policy (fresh normal latent and conditional Gaussian noise)
with at least200 episodes per task. Preserve raw trajectories, full initial
states, source/checkpoint hashes and route counts. Route labels indicate gate
entry, not successful goal reaching. The replay audit's coordinate occupancy
is not a conditional action-mode count. Do not restart training or modify
checkpoints, frozen sources, W&B runs, or original evaluation files.
