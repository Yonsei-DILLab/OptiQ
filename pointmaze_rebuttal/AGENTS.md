# PointMaze rebuttal boundary

This entire folder is PointMaze-only and must never be copied, committed,
merged, cherry-picked, pushed or exported to any anonymous branch or mirror.
The repository-root AGENTS.md is mandatory. Keep new experiment code, profiles,
tests and documentation here; keep generated files under ignored outputs/.
Frozen environment and iBOLT core copies are kept inside this folder. Do not
import or modify the root pointmaze/, ibolt/, models/, configs/ or gmm40/ code
to implement this experiment. SOURCE.json records the snapshot and adaptation.
Preserve vendor/dipo source bytes and license; SOURCE.json pins their hashes.

## Primary iBOLT reporting basis

For PointMaze success, goal coverage, goal distributions and their figures/tables,
use `mu_only` as the primary evaluation: draw a fresh stochastic latent z at every
step and execute the conditional mean, without conditional Gaussian noise.
This is not fixed z=0. Keep full-policy evaluation separately labeled as
supplementary; never mix its numbers into the primary comparison. Training-reward
discovery statistics continue to use the actual training history. This reporting
rule does not change training, checkpoints, or other experiment families.
